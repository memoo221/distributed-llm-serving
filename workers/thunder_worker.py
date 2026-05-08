"""GPU worker that runs on a Thunder Compute instance.

Same contract as the existing CPU worker (workers/worker_router.py):
- POST /generate {prompt, max_new_tokens} -> {response}
- Heartbeats to MASTER_URL/heartbeat with active_requests so the master can
  load-balance.

Self-contained — no imports from the rest of the project — so you can scp
just this file to Thunder if you don't want to ship the whole repo.

Run on Thunder:
    pip install -r thunder_requirements.txt
    MASTER_URL=https://<your-tunnel>.trycloudflare.com \
    SELF_URL=https://<uuid>-<port>.thundercompute.net \
    WORKER_ID=thunder_a_gpu0 \
    WORKER_DEVICE=cuda:0 \
    uvicorn thunder_worker:app --host 0.0.0.0 --port 8000

WORKER_DEVICE pins the worker to a specific GPU on multi-GPU instances.
Required on Thunder Compute because their CUDA runtime intercepts
CUDA_VISIBLE_DEVICES and routes everything to the first physical GPU.
Use cuda:0 / cuda:1 / etc. to select; omit to default to "cuda".
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager

import httpx
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
WORKER_ID = os.getenv("WORKER_ID", "thunder_worker_1")
SELF_URL = os.getenv("SELF_URL", "http://localhost:8000")
MASTER_URL = os.getenv("MASTER_URL", "")
HEARTBEAT_SEC = float(os.getenv("HEARTBEAT_INTERVAL", "5"))
WORKER_API_KEY = os.getenv("WORKER_API_KEY", "")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

# Filled in lifespan; None until the model finishes loading.
_tokenizer = None
_model = None
# Allow WORKER_DEVICE to override (e.g. "cuda:0" or "cuda:1") for multi-GPU
# instances where CUDA_VISIBLE_DEVICES is intercepted by the host runtime.
_device = os.getenv("WORKER_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

# Heartbeat counter. Inference runs in a thread (asyncio.to_thread), so this
# is touched from multiple threads — guard with a Lock. The asyncio path
# could use a plain int, but consistency is cheaper than auditing later.
_active = 0
_active_lock = threading.Lock()


def _inc() -> None:
    global _active
    with _active_lock:
        _active += 1


def _dec() -> None:
    global _active
    with _active_lock:
        _active -= 1


def _snapshot_active() -> int:
    with _active_lock:
        return _active


async def heartbeat_loop() -> None:
    if not MASTER_URL:
        print(f"[{WORKER_ID}] MASTER_URL not set, skipping heartbeats")
        return

    headers = {"X-API-Key": WORKER_API_KEY} if WORKER_API_KEY else {}
    async with httpx.AsyncClient(timeout=5.0) as http:
        while True:
            payload = {
                "worker_id": WORKER_ID,
                "url": SELF_URL,
                "device_type": "gpu",
                "active_requests": _snapshot_active(),
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
    global _tokenizer, _model
    print(f"[{WORKER_ID}] loading {MODEL_NAME} on {_device}...")
    t0 = time.time()
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dtype = torch.float16 if _device.startswith("cuda") else torch.float32
    _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    _model = _model.to(_device)
    _model.eval()
    if _device.startswith("cuda"):
        device_id = int(_device.split(":", 1)[1]) if ":" in _device else None
        free, total = torch.cuda.mem_get_info(device_id)
        used_mb = (total - free) / (1024 * 1024)
        print(f"[{WORKER_ID}] loaded in {time.time()-t0:.1f}s, device={_device}, vram_used={used_mb:.0f}MB")
    else:
        print(f"[{WORKER_ID}] loaded in {time.time()-t0:.1f}s (cpu fallback)")
    print(f"[{WORKER_ID}] master={MASTER_URL or 'none'}, self_url={SELF_URL}")
    task = asyncio.create_task(heartbeat_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(lifespan=lifespan)


class Request(BaseModel):
    prompt: str
    max_new_tokens: int = 256


@app.get("/health")
async def health():
    return {
        "status": "ok" if _model is not None else "loading",
        "worker_id": WORKER_ID,
        "device": _device,
        "model": MODEL_NAME,
        "active_requests": _snapshot_active(),
    }


@app.post("/generate")
async def generate(req: Request):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    _inc()
    try:
        # transformers.generate is blocking; offload to a thread so the event
        # loop stays free for heartbeats and incoming HTTP. Multiple concurrent
        # /generate calls will serialize on the GPU (no batching), which is
        # fine for a single-stream worker — the master schedules around it.
        text = await asyncio.to_thread(_run_inference, req.prompt, req.max_new_tokens)
        return {"response": text}
    except torch.cuda.OutOfMemoryError as e:
        # Free the cached blocks so the next request has a chance to fit.
        torch.cuda.empty_cache()
        raise HTTPException(status_code=503, detail=f"cuda oom: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    finally:
        _dec()


def _run_inference(prompt: str, max_new_tokens: int) -> str:
    # Apply the model's chat template when available so instruct-tuned models
    # behave correctly. Falls back to raw prompt for base models.
    messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer clearly and concisely in English."},
                {"role": "user", "content": prompt},
            ]
    if hasattr(_tokenizer, "apply_chat_template") and _tokenizer.chat_template:
        text = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt

    inputs = _tokenizer(text, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=_tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
