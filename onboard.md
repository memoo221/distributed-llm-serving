# Distributed LLM Inference — Onboarding

## Overview

This project implements a distributed system for handling concurrent LLM inference requests with load balancing, GPU task distribution, and fault tolerance.

Current request flow:

```
Client → NGINX:8008 → Master Nodes:7000 → Worker Nodes
```

---

## NGINX Load Balancer + Mock Masters (Docker)

This local Docker stack validates NGINX load balancing across 4 mock master nodes before the real master-node logic is implemented.

### What's included

- NGINX as the public entry point on port `8008`
- 4 simulated master nodes: `master1`, `master2`, `master3`, `master4`
- Each mock master runs a lightweight FastAPI service on port `7000`
- A traffic simulation script to verify request distribution across masters
- Mock master routes refactored into `master/routers/mock_api.py`

### Files

- `docker-compose.yml`
- `nginx/nginx.conf`
- `master/Dockerfile.mock`
- `master/mock_api.py`
- `master/routers/mock_api.py`
- `tests/simulate_nginx_lb.py`
- `.dockerignore`

### How to run

1. Start Docker Desktop.
2. From the project root:

```powershell
docker compose up --build -d
```

3. Confirm all containers are up:

```powershell
docker compose ps -a
```

Expected:
- `distributed-nginx` is `Up`
- `master1`, `master2`, `master3`, `master4` are all `Up`

### How to verify

**1. NGINX health:**

```powershell
curl http://localhost:8008/nginx/health
```

Expected response:

```text
nginx ok
```

**2. Requests are forwarded to a master:**

```powershell
curl http://localhost:8008/
curl http://localhost:8008/health
curl -X POST http://localhost:8008/generate -H "Content-Type: application/json" -d "{\"prompt\":\"hello\",\"delay_ms\":500}"
```

Expected:
- HTTP `200`
- JSON response containing `master_id`

**3. Verify load balancing:**

```powershell
python tests/simulate_nginx_lb.py --requests 20 --concurrency 8 --delay-ms 1000
```

Expected:
- All requests return `status=200`
- Final summary shows traffic distributed across all 4 masters

```text
master1: 5
master2: 5
master3: 5
master4: 5
```

### Failover check

Validate that NGINX reroutes traffic when a master is unavailable:

```powershell
docker compose stop master2
python tests/simulate_nginx_lb.py --requests 20 --concurrency 8 --delay-ms 1000
docker compose start master2
```

Expected:
- Requests still succeed while `master2` is stopped
- `master2` disappears from the distribution summary during the stop window

### Important ports

- `8008` — NGINX load balancer
- `7000` — mock master node services

---

## LLM Models Setup

### Download models

Models are stored locally and are **not pushed to GitHub** (excluded via `.gitignore`).

**TinyLlama 1.1B Chat** (~2.2 GB):

```powershell
python models/download_model.py
```

Saves to: `models/tinyllama-1.1b-chat/`

**Qwen 2.5 0.5B** (~1 GB):

```powershell
python models/download_qwen.py
```

Saves to: `models/qwen2.5-0.5b/`

> Both scripts resume automatically if the download is interrupted.

---

### Starting the worker server

The worker exposes a `/generate` endpoint that accepts a question and returns an LLM response.

**Default (TinyLlama):**

```powershell
uvicorn workers.worker_router:app --host 0.0.0.0 --port 9001 --reload
```

**With Qwen instead:**

```powershell
$env:MODEL_PATH="models/qwen2.5-0.5b"; uvicorn workers.worker_router:app --host 0.0.0.0 --port 9001 --reload
```

**Test the endpoint:**

```powershell
curl -X POST http://localhost:9001/generate `
  -H "Content-Type: application/json" `
  -d '{\"question\": \"What is the capital of France?\"}'
```

Expected response:

```json
{
  "question": "What is the capital of France?",
  "answer": "The capital of France is Paris."
}
```

**Health check:**

```powershell
curl http://localhost:9001/health
```

> On CPU, responses take 30–120 seconds. On a CUDA GPU, install:
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`
> and the model will run in 1–3 seconds automatically.

---

## GPU Worker Nodes (Kaggle)

Two real GPU worker nodes run on Kaggle Tesla T4s, exposed via private Cloudflare Tunnel. They serve real LLM inference and plug into the master tier as remote workers.

### Endpoints

| Worker | Hostname | GPU | Model |
|---|---|---|---|
| `worker_1.1` | `$WORKER1_URL` | Tesla T4 (16 GB) | Qwen 2.5 0.5B Instruct |
| `worker_1.2` | `$WORKER2_URL` | Tesla T4 (16 GB) | Qwen 2.5 0.5B Instruct |

Actual hostnames live in `.env` / team password manager. **Do not hardcode them in source or docs.**

### Required environment variables

Copy `.env.example` to `.env` and fill values from the team vault:

```
WORKER1_URL=
WORKER2_URL=
WORKER_API_KEY=
```

All `/generate` calls require the header:

```
X-API-Key: $WORKER_API_KEY
```

### API

Each GPU worker exposes:

- `GET /health` — status + GPU stats (no auth)
- `GET /stats` — same payload, requires `X-API-Key`
- `POST /generate` — LLM inference, requires `X-API-Key`

### Generate example

```powershell
curl -X POST $env:WORKER1_URL/generate `
  -H "Content-Type: application/json" `
  -H "X-API-Key: $env:WORKER_API_KEY" `
  -d '{\"question\": \"What is distributed computing?\"}'
```

### Worker capabilities

- **Dynamic batching** (50ms window, batch size 8) — measured 13× throughput vs. single-request mode (16 → 212 tokens/sec).
- **GPU telemetry** in every response: per-request memory, output tokens, batch size, tokens/sec. `/health` adds GPU utilization %, VRAM, temperature.
- **Load-tested**: 256 concurrent requests with zero errors. Sweet spot ~16 concurrent per worker, ~3.5 req/sec sustained.

### Session constraints

- Kaggle sessions cap at ~9h wall-clock and idle out after ~20min.
- Tunnel hostnames are fixed — URLs persist across kernel restarts.
- If a GPU worker returns 502/504, the Kaggle session has died. Ping the worker maintainer to restart.

### Per-worker limits

| Metric | Value |
|---|---|
| Sweet spot concurrent | 16 |
| Sustained throughput | 3.5 req/sec (32-token outputs) |
| Hard ceiling | 256+ concurrent (no errors, high latency) |
| Two-worker total | ~7 req/sec, ~32 concurrent |

---

## Worker API Contract

All worker nodes (local CPU, Kaggle GPU, mock) MUST expose the same API so the master can route to any of them interchangeably.

### `GET /health`

Returns worker status. No authentication.

```json
{
  "status": "ok",
  "worker_id": "worker_1.1",
  "active_requests": 0,
  "uptime_sec": 123.4
}
```

Optional fields when available: `gpu_util_pct`, `vram_used_mb`, `queue_depth`, `total_requests`, `total_errors`.

### `POST /generate`

Runs inference. Requires `X-API-Key` header on remote workers.

**Request:**

```json
{
  "question": "What is distributed computing?",
  "max_new_tokens": 256
}
```

**Response:**

```json
{
  "worker_id": "worker_1.1",
  "response": "Distributed computing refers to...",
  "stats": {
    "latency_ms": 528.6,
    "output_tokens": 70
  }
}
```

`stats` may include additional fields when available: `batch_size`, `peak_mem_mb`, `per_request_mem_mb`, `tokens_per_sec`.

---

## Optional local backend run

To run the separate backend app manually outside the Docker mock stack:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

This is separate from the NGINX simulation and should not be run on port `8008`.

---

## Notes

- `nginx/nginx.conf` is mounted into the container, so config edits do not require rebuilding the image.
- Python code inside the mock master image does require rebuilds:

```powershell
docker compose up --build -d
```

- `.env` must never be committed. `.env.example` documents the required variables.

---

## Current Scope

- ✅ NGINX load balancer with 4 mock masters (Docker)
- ✅ Worker server skeleton (`workers/worker_router.py`)
- ✅ Two real GPU workers on Kaggle (Tesla T4)
- ⏳ Real master-node scheduling and worker registry
- ⏳ Heartbeat-based health monitoring on the master
- ⏳ Load-aware routing across heterogeneous workers (GPU + CPU)
- ⏳ End-to-end fault tolerance demo
