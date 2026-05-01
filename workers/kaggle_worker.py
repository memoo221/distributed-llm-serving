import os
import sys
import time
import asyncio
from collections import deque
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from inference import load_model, generate_response, generate_batch, get_gpu_stats

# ---------- Config (from CLI args + env vars) ----------
WORKER_ID  = sys.argv[1]                                  # e.g. "worker_1.1"
DEVICE     = sys.argv[2]                                  # e.g. "cuda:0"
PORT       = int(sys.argv[3])                             # e.g. 8000

MODEL_PATH      = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
API_KEY         = os.environ.get("WORKER_API_KEY", "")
MASTER_URL      = os.environ.get("MASTER_URL", "")        # empty => skip heartbeats
SELF_URL        = os.environ.get(f"{WORKER_ID.upper().replace('.', '_')}_PUBLIC_URL", "")
HEARTBEAT_SEC   = float(os.environ.get("HEARTBEAT_INTERVAL", "5"))

# Batching config
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE", "8"))
BATCH_WINDOW_MS = int(os.environ.get("BATCH_WINDOW_MS", "50"))
USE_BATCHING    = os.environ.get("USE_BATCHING", "1") == "1"

# ---------- State ----------
active_requests = 0
total_requests  = 0
total_errors    = 0
started_at      = time.time()

# Pending request queue for the batcher
_queue: deque = deque()
_batch_event: asyncio.Event = None  # created on startup


# ---------- Batching loop ----------
async def batch_loop():
    """Collect requests within a small window and process them as one GPU batch."""
    global total_requests, total_errors

    while True:
        await _batch_event.wait()
        # Wait for the batch window so more requests can join
        await asyncio.sleep(BATCH_WINDOW_MS / 1000)

        # Drain up to BATCH_SIZE items from the queue
        batch = []
        while _queue and len(batch) < BATCH_SIZE:
            batch.append(_queue.popleft())
        if not _queue:
            _batch_event.clear()
        if not batch:
            continue

        questions     = [item["question"]       for item in batch]
        max_tokens    = max(item["max_new_tokens"] for item in batch)

        try:
            results = await run_in_threadpool(generate_batch, questions, max_tokens)
            for item, res in zip(batch, results):
                item["future"].set_result(res)
                total_requests += 1
        except Exception as e:
            total_errors += len(batch)
            for item in batch:
                if not item["future"].done():
                    item["future"].set_exception(e)


# ---------- Heartbeat loop ----------
async def heartbeat_loop():
    """POST status to the master every HEARTBEAT_SEC seconds."""
    if not MASTER_URL:
        print("[heartbeat] MASTER_URL not set, skipping heartbeats")
        return

    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            payload = {
                "worker_id": WORKER_ID,
                "url": SELF_URL,
                "active_requests": active_requests,
                "total_requests": total_requests,
                "total_errors": total_errors,
                "queue_depth": len(_queue),
                "uptime_sec": round(time.time() - started_at, 1),
                "timestamp": time.time(),
                **get_gpu_stats(),
            }
            try:
                await client.post(f"{MASTER_URL}/heartbeat", json=payload, headers=headers)
            except Exception as e:
                print(f"[heartbeat] failed: {e}")
            await asyncio.sleep(HEARTBEAT_SEC)


# ---------- FastAPI lifespan: load model + start background tasks ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _batch_event
    print(f"[{WORKER_ID}] Loading model on {DEVICE}...")
    load_model(MODEL_PATH, device=DEVICE)
    print(f"[{WORKER_ID}] Model loaded.")

    _batch_event = asyncio.Event()
    tasks = []
    if USE_BATCHING:
        tasks.append(asyncio.create_task(batch_loop()))
        print(f"[{WORKER_ID}] Batching enabled (size={BATCH_SIZE}, window={BATCH_WINDOW_MS}ms)")
    if MASTER_URL:
        tasks.append(asyncio.create_task(heartbeat_loop()))
        print(f"[{WORKER_ID}] Heartbeats -> {MASTER_URL} every {HEARTBEAT_SEC}s")

    yield

    for t in tasks:
        t.cancel()


app = FastAPI(lifespan=lifespan)


# ---------- Auth helper ----------
def _check_key(x_api_key: str):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "invalid api key")


# ---------- Schemas ----------
class GenerateReq(BaseModel):
    question: str
    max_new_tokens: int = 256


# ---------- Endpoints ----------
@app.get("/health")
def health(x_api_key: str = Header(None)):
    # Health is open by design so the master can ping it without a key,
    # but you can uncomment this to require one:
    # _check_key(x_api_key)
    return {
        "status": "ok",
        "worker_id": WORKER_ID,
        "active_requests": active_requests,
        "total_requests": total_requests,
        "total_errors": total_errors,
        "queue_depth": len(_queue),
        "uptime_sec": round(time.time() - started_at, 1),
        "batching": USE_BATCHING,
        "batch_size": BATCH_SIZE,
        **get_gpu_stats(),
    }


@app.get("/stats")
def stats(x_api_key: str = Header(None)):
    _check_key(x_api_key)
    return {
        "worker_id": WORKER_ID,
        "active_requests": active_requests,
        "total_requests": total_requests,
        "total_errors": total_errors,
        "queue_depth": len(_queue),
        "uptime_sec": round(time.time() - started_at, 1),
        **get_gpu_stats(),
    }


@app.post("/generate")
async def generate(r: GenerateReq, x_api_key: str = Header(None)):
    _check_key(x_api_key)
    global active_requests, total_requests, total_errors
    active_requests += 1
    try:
        if USE_BATCHING:
            # Hand off to the batcher and wait for our result
            fut: asyncio.Future = asyncio.get_event_loop().create_future()
            _queue.append({
                "question": r.question,
                "max_new_tokens": r.max_new_tokens,
                "future": fut,
            })
            _batch_event.set()
            result = await fut
        else:
            # Direct (non-batched) path
            result = await run_in_threadpool(generate_response, r.question, r.max_new_tokens)
            total_requests += 1
        return {"worker_id": WORKER_ID, **result}
    except Exception as e:
        total_errors += 1
        raise HTTPException(500, f"generation failed: {e}")
    finally:
        active_requests -= 1


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
