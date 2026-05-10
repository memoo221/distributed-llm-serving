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
from dataclasses import dataclass, field

import httpx
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


async def _sample_gpu_stats(gpu_index: int) -> tuple[float | None, float | None]:
    """Shell out to nvidia-smi for util% and used VRAM (MB) on a single GPU.

    We use the subprocess (not pynvml) because Thunder Compute's virtualized
    CUDA layer is known to interfere with low-level NVML calls but
    `nvidia-smi --query-gpu=...` works reliably on those instances. Returns
    (None, None) if nvidia-smi is missing or the call fails for any reason —
    the worker stays healthy, the UI just shows "—" for that field.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
        line = out.decode().strip().splitlines()[0]
        util_str, mem_str = [x.strip() for x in line.split(",")]
        return float(util_str), float(mem_str)
    except Exception:
        return None, None


# Use `or default` instead of os.getenv(key, default) — the latter returns ""
# when the var is set-but-empty, which has bitten us when launch_workers.sh
# accidentally let MODEL_NAME='' through. `or` falls back to default for
# both unset AND empty values.
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
WORKER_ID = os.getenv("WORKER_ID") or "thunder_worker_1"
SELF_URL = os.getenv("SELF_URL") or "http://localhost:8000"
MASTER_URL = os.getenv("MASTER_URL", "")
HEARTBEAT_SEC = float(os.getenv("HEARTBEAT_INTERVAL") or "5")
WORKER_API_KEY = os.getenv("WORKER_API_KEY", "")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

# Continuous-batching knobs. The batcher coalesces concurrent /generate calls
# into a single model.generate() so multiple prompts share each forward pass.
# BATCH_SIZE caps how many requests run together; BATCH_WAIT_MS is how long
# to wait for more requests to arrive after the first one before flushing.
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
BATCH_WAIT_MS = int(os.getenv("BATCH_WAIT_MS", "30"))

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


@dataclass
class _PendingRequest:
    prompt: str
    max_new_tokens: int
    future: asyncio.Future = field(default=None)


# Set in lifespan once the asyncio loop is running.
_request_queue: asyncio.Queue | None = None


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

    gpu_index = None
    if _device.startswith("cuda"):
        gpu_index = int(_device.split(":", 1)[1]) if ":" in _device else 0

    async with httpx.AsyncClient(timeout=5.0) as http:
        while True:
            gpu_util_pct = None
            vram_used_mb = None
            if gpu_index is not None:
                gpu_util_pct, vram_used_mb = await _sample_gpu_stats(gpu_index)

            payload = {
                "worker_id": WORKER_ID,
                "url": SELF_URL,
                "device_type": "gpu",
                "active_requests": _snapshot_active(),
                "queue_depth": 0,
                "slots": BATCH_SIZE,
                "gpu_util_pct": gpu_util_pct,
                "vram_used_mb": vram_used_mb,
                "timestamp": time.time(),
            }
            try:
                await http.post(f"{MASTER_URL}/heartbeat", json=payload, headers=headers)
            except Exception as e:
                print(f"[{WORKER_ID}] heartbeat failed: {type(e).__name__}: {e}")
            await asyncio.sleep(HEARTBEAT_SEC)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tokenizer, _model, _request_queue
    print(f"[{WORKER_ID}] loading {MODEL_NAME} on {_device}...")
    t0 = time.time()
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Causal LMs need left-padding so the generated tokens come out at the
    # right side of every sequence in a batch.
    _tokenizer.padding_side = "left"
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token = _tokenizer.eos_token
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
    print(f"[{WORKER_ID}] batching enabled: BATCH_SIZE={BATCH_SIZE}, BATCH_WAIT_MS={BATCH_WAIT_MS}")
    _request_queue = asyncio.Queue()
    hb_task = asyncio.create_task(heartbeat_loop())
    batch_task = asyncio.create_task(_batcher_loop())
    try:
        yield
    finally:
        hb_task.cancel()
        batch_task.cancel()


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
    if _model is None or _request_queue is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    pending = _PendingRequest(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        future=asyncio.get_running_loop().create_future(),
    )
    _inc()
    try:
        await _request_queue.put(pending)
        text = await pending.future
        return {"response": text}
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=503, detail=f"cuda oom: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    finally:
        _dec()


async def _batcher_loop() -> None:
    # Single consumer of _request_queue. Coalesces concurrent requests into
    # one batched model.generate() per iteration so multiple prompts share
    # the same forward passes on the GPU.
    assert _request_queue is not None
    wait_s = BATCH_WAIT_MS / 1000.0
    while True:
        first = await _request_queue.get()
        batch: list[_PendingRequest] = [first]
        deadline = time.monotonic() + wait_s
        while len(batch) < BATCH_SIZE:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                nxt = await asyncio.wait_for(_request_queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                break
            batch.append(nxt)

        try:
            results = await asyncio.to_thread(_run_batch, batch)
            for req, text in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(text)
        except Exception as e:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)


def _format_prompt(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer clearly and concisely in English."},
        {"role": "user", "content": prompt},
    ]
    if hasattr(_tokenizer, "apply_chat_template") and _tokenizer.chat_template:
        return _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def _run_batch(batch: list[_PendingRequest]) -> list[str]:
    prompts = [_format_prompt(b.prompt) for b in batch]
    # All sequences in a batch share the same generation length budget — use
    # the largest requested so no single request gets truncated. Per-sequence
    # EOS still stops earlier finishers naturally inside generate().
    max_new = max(b.max_new_tokens for b in batch)

    inputs = _tokenizer(prompts, return_tensors="pt", padding=True).to(_device)
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=_tokenizer.pad_token_id,
        )
    input_len = inputs.input_ids.shape[1]
    results: list[str] = []
    for i in range(len(batch)):
        new_tokens = outputs[i][input_len:]
        text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(text)
    return results
