# Running GPU Workers on Kaggle

This document explains how to spin up `kaggle_worker.py` on Kaggle's free 2× T4 GPUs and expose them publicly via ngrok. The notebook itself is **not committed** to this repo (Kaggle-specific, depends on per-account secrets). Follow these steps to set up your own.

## Prerequisites

- Kaggle account with phone verification (required to enable GPUs)
- ngrok account (free at https://ngrok.com)
- This repo's `WORKER_API_KEY` from the team vault

## Kaggle Notebook Settings

1. Create a new notebook on kaggle.com → **Create → New Notebook**
2. Right sidebar:
   - **Accelerator** → `GPU T4 x2`
   - **Internet** → `On`

## Kaggle Secrets

Top menu → **Add-ons → Secrets**. Add:

| Name | Value |
|---|---|
| `WORKER_API_KEY` | from team vault |
| `NGROK_TOKEN` | from https://dashboard.ngrok.com/get-started/your-authtoken |

## Notebook Cells

### Cell 1 — Install dependencies

```python
!pip install -q fastapi uvicorn nest_asyncio transformers accelerate pynvml pyngrok httpx
```

### Cell 2 — Clone this repo

```python
!git clone https://github.com/<yourorg>/<yourrepo>.git
%cd <yourrepo>/workers
```

### Cell 3 — Pre-download model (one-time per session)

```python
import os
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
AutoTokenizer.from_pretrained(MODEL)
AutoModelForCausalLM.from_pretrained(MODEL)
print("Model cached.")
```

### Cell 4 — Launch the two workers

```python
import os, subprocess, time
from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()

env = {
    **os.environ,
    "HF_HOME": "/kaggle/working/hf_cache",
    "WORKER_API_KEY": s.get_secret("WORKER_API_KEY"),
    "USE_BATCHING": "1",
    "BATCH_SIZE": "8",
    "BATCH_WINDOW_MS": "50",
    "PYTHONUNBUFFERED": "1",
}

w1_log = open("/kaggle/working/w1.log", "w")
w2_log = open("/kaggle/working/w2.log", "w")
w1 = subprocess.Popen(["python", "-u", "kaggle_worker.py", "worker_1.1", "cuda:0", "8000"],
                     env=env, stdout=w1_log, stderr=subprocess.STDOUT)
w2 = subprocess.Popen(["python", "-u", "kaggle_worker.py", "worker_1.2", "cuda:1", "8001"],
                     env=env, stdout=w2_log, stderr=subprocess.STDOUT)

print("Waiting 30s for workers...")
time.sleep(30)
print("Done.")
```

### Cell 5 — Sanity check

```python
import requests
print(requests.get("http://localhost:8000/health").json())
print(requests.get("http://localhost:8001/health").json())
```

Both should return JSON with `"status": "ok"` and GPU stats.

### Cell 6 — Expose publicly via ngrok proxy

Free ngrok allows only one tunnel, so we run a tiny FastAPI proxy on port 7000 that fronts both workers with path-based routing (`/w1/*` → port 8000, `/w2/*` → port 8001), then expose only the proxy.

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx, threading, uvicorn, nest_asyncio
from pyngrok import ngrok
from kaggle_secrets import UserSecretsClient

s = UserSecretsClient()
ngrok.set_auth_token(s.get_secret("NGROK_TOKEN"))

for t in ngrok.get_tunnels():
    ngrok.disconnect(t.public_url)
ngrok.kill()

proxy = FastAPI()
client = httpx.AsyncClient(timeout=120)
WORKERS = {"w1": "http://localhost:8000", "w2": "http://localhost:8001"}

@proxy.api_route("/{worker}/{path:path}", methods=["GET", "POST"])
async def route(worker: str, path: str, request: Request):
    if worker not in WORKERS:
        return JSONResponse({"error": f"unknown worker '{worker}'"}, status_code=404)
    body = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}
    r = await client.request(request.method, f"{WORKERS[worker]}/{path}",
                             content=body, headers=headers)
    return JSONResponse(r.json(), status_code=r.status_code)

nest_asyncio.apply()
threading.Thread(
    target=lambda: uvicorn.run(proxy, host="0.0.0.0", port=7000, log_level="warning"),
    daemon=True,
).start()

public = ngrok.connect(7000, "http").public_url
print(f"\nPublic URL: {public}")
print(f"  Worker 1.1: {public}/w1")
print(f"  Worker 1.2: {public}/w2")
```

Paste the printed URLs into the master's worker config:

```json
{
  "workers": [
    {"id": "worker_1.1", "url": "<public-url>/w1"},
    {"id": "worker_1.2", "url": "<public-url>/w2"}
  ]
}
```

## Verification (from your laptop)

```powershell
curl <public-url>/w1/health
curl <public-url>/w2/health

curl -X POST <public-url>/w1/generate `
  -H "Content-Type: application/json" `
  -H "X-API-Key: $env:WORKER_API_KEY" `
  -d '{\"question\": \"What is distributed computing?\"}'
```

## Notes

- ngrok URLs are **not stable** on the free tier — they change every Kaggle session. Refresh the master's worker config after each restart.
- Kaggle sessions cap at ~9 hours wall-clock and idle out after ~20 minutes. Don't close the browser tab during a load test.
- If a worker returns 502, the Kaggle kernel has died. Restart the notebook from Cell 1.
- Logs are written to `/kaggle/working/w1.log` and `w2.log`. Inspect with `!cat /kaggle/working/w1.log` if a worker fails to start.

## Why a proxy instead of two tunnels?

Free ngrok caps at one simultaneous HTTP tunnel per account. Running a tiny FastAPI proxy in front of both workers and exposing only the proxy keeps us on the free tier. The proxy adds ~1ms per request (negligible vs. ~600ms inference) and mirrors the role of NGINX in the master tier — Layer 7 routing by URL path.
