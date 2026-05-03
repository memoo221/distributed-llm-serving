import asyncio
import os
import time
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from groq import Groq
from pydantic import BaseModel

load_dotenv()

WORKER_ID      = os.getenv("WORKER_ID", "groq_worker")
SELF_URL       = os.getenv("SELF_URL", "")
MASTER_URL     = os.getenv("MASTER_URL", "")
HEARTBEAT_SEC  = float(os.getenv("HEARTBEAT_INTERVAL", "5"))
WORKER_API_KEY = os.getenv("WORKER_API_KEY", "")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Live counter — incremented on entry to /generate, decremented in finally.
# The master uses this (via heartbeat) to balance load across workers.
active_requests = 0


async def heartbeat_loop() -> None:
    """POST {worker_id, url, device_type, active_requests, queue_depth, timestamp}
    to MASTER_URL/heartbeat every HEARTBEAT_SEC seconds. Failures are logged,
    not raised — a missed heartbeat just means the master will mark us stale
    after WORKER_STALE_SEC and stop routing to us until we recover."""
    if not MASTER_URL:
        print(f"[{WORKER_ID}] MASTER_URL not set — skipping heartbeats")
        return

    headers = {"X-API-Key": WORKER_API_KEY} if WORKER_API_KEY else {}
    async with httpx.AsyncClient(timeout=5.0) as http:
        while True:
            payload = {
                "worker_id": WORKER_ID,
                "url": SELF_URL,
                "device_type": "gpu",
                "active_requests": active_requests,
                "queue_depth": 0,
                "timestamp": time.time(),
            }
            try:
                await http.post(f"{MASTER_URL}/heartbeat", json=payload, headers=headers)
            except Exception as e:
                print(f"[{WORKER_ID}] heartbeat failed: {type(e).__name__}: {e}")
            await asyncio.sleep(HEARTBEAT_SEC)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[{WORKER_ID}] starting; master={MASTER_URL or 'none'}, self_url={SELF_URL or 'none'}")
    task = asyncio.create_task(heartbeat_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


class Request(BaseModel):
    prompt: str
    max_new_tokens: int = 256


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "worker_id": WORKER_ID,
        "active_requests": active_requests,
    }


@app.post("/generate")
async def generate(req: Request):
    global active_requests
    active_requests += 1
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer clearly and concisely in English."},
                {"role": "user", "content": req.prompt},
            ],
            max_completion_tokens=req.max_new_tokens,
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}
    finally:
        active_requests -= 1
