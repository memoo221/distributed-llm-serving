# Distributed LLM Inference - Onboarding

## Overview

This project implements a distributed system for concurrent LLM inference with load balancing, scheduler-driven worker selection, heartbeat-based health monitoring, and retry/failover behavior.

Current request flow:

```text
Client -> NGINX:8008 -> Master Nodes:7000 -> Worker Nodes (local CPU + Modal-backed GPU)
```

Workers are heterogeneous:

- CPU workers load Qwen 2.5 0.5B locally and serve from a single in-process slot.
- Modal workers are thin FastAPI shims that forward requests to a deployed Modal endpoint backed by `Qwen/Qwen2.5-Coder-7B-Instruct`. They count as `device_type: "gpu"` for scheduling.

The master's scheduler does threshold-based routing: requests with an estimated token score >= `256` go GPU-only; smaller requests prefer CPU via JSQ(2), with fallback to GPU when the CPU pool saturates.

---

## Stack

The local Docker Compose stack runs the full end-to-end pipeline.

### What's Included

- NGINX as the public entry point on port `8008` (least-conn load balancing across masters)
- 2 master nodes: `master1`, `master2` (FastAPI on internal port `7000`; mapped to host `7001` and `7002`)
- 2 CPU workers: `cpu_worker_1_1`, `cpu_worker_2_1` (one per master, internal port `9001`)
- 2 Modal worker shims: `modal_worker_1_2`, `modal_worker_2_2` (one per master, internal port `8000`)

### Files

- `docker-compose.yml`
- `nginx/nginx.conf`
- `master/Dockerfile.master_node`, `master/master_app.py`, `master/services/*`, `master/routers/master_router.py`
- `workers/Dockerfile.cpu`, `workers/worker_router.py`, `workers/worker_service.py` - local CPU pipeline
- `workers/Dockerfile.groq`, `workers/worker_modal.py` - Modal-backed GPU shim
- `tests/simulate_nginx_lb.py` - NGINX-level master LB test
- `tests/simulate_worker_scheduling.py` - end-to-end scheduler test
- `tests/smoke_worker_node.py` - direct smoke test of `WorkerNode`

### Required Environment Variables

Typical values for the current topology:

```env
MODAL_WORKER_URL=https://your-modal-endpoint.modal.run
MODAL_API_KEY=
WORKER_API_KEY=

GPU_SLOTS=1

MODAL_MIN_CONTAINERS=2
MODAL_MAX_CONTAINERS=2
MODAL_BUFFER_CONTAINERS=0
MODAL_TARGET_INPUTS=1
MODAL_MAX_INPUTS=1
MODAL_SCALEDOWN_WINDOW=300
```

Notes:

- `MODAL_WORKER_URL` is the deployed Modal endpoint used by both local modal worker shims.
- If both shims point to the same URL, they share one Modal backend pool.
- `GPU_SLOTS` is the master's scheduling assumption, not a direct Modal autoscaling control.

### How to Run

```powershell
docker compose up --build -d
docker compose ps -a
```

Expected `Up`:

- `distributed-nginx`
- `master1`
- `master2`
- `cpu_worker_1_1`
- `cpu_worker_2_1`
- `modal_worker_1_2`
- `modal_worker_2_2`

---

## Verification

### 1. NGINX Health

```powershell
curl http://localhost:8008/nginx/health
```

### 2. Each Master Sees Its 2 Workers

```powershell
curl http://localhost:7001/scheduler/workers
curl http://localhost:7002/scheduler/workers
```

Each master should show:

- 1 CPU worker
- 1 GPU worker

`device_type` should be `cpu` for one and `gpu` for the other.

### 3. End-to-End via NGINX

```powershell
# small request (max_new_tokens < 256) -> usually routes to CPU
curl -X POST http://localhost:8008/generate -H "Content-Type: application/json" -d '{"prompt":"hello","max_new_tokens":32}'

# large request (max_new_tokens >= 256) -> routes to a modal worker
curl -X POST http://localhost:8008/generate -H "Content-Type: application/json" -d '{"prompt":"hello","max_new_tokens":300}'
```

Responses should include:

- `master_id`
- `worker_id`
- `response`

When the request lands on the Modal path, you may also see Modal-specific fields like:

- `modal_task_id`
- `modal_endpoint_task_id`
- `modal_region`
- `modal_model`

---

## Test Scripts

### NGINX Load-Balancing Check

```powershell
python tests/simulate_nginx_lb.py --base-url http://localhost:8008 --requests 30 --concurrency 8
```

### Worker Scheduling Check

```powershell
# all large -> should route to modal workers
python tests/simulate_worker_scheduling.py --base-url http://localhost:8008 --requests 20 --large-ratio 1

# all small -> should mainly route to CPU workers
python tests/simulate_worker_scheduling.py --base-url http://localhost:8008 --requests 20 --large-ratio 0

# mixed workload
python tests/simulate_worker_scheduling.py --base-url http://localhost:8008 --requests 30 --large-ratio 0.3 --concurrency 8
```

Expected distribution by worker:

| Run | Workers you should see |
|---|---|
| `--large-ratio 1` | `modal_worker_1_2` + `modal_worker_2_2` |
| `--large-ratio 0` | `worker_1_1` + `worker_2_1`, with possible GPU fallback under saturation |
| `--large-ratio 0.3` | mix of all four |

The script reports:

- per-request routing
- latency stats
- throughput
- CPU/GPU utilization estimates
- counts for `small -> GPU` fallback
- counts for `large -> CPU` misroutes

### Failover Check

```powershell
docker compose stop master1
python tests/simulate_worker_scheduling.py --base-url http://localhost:8008 --requests 10 --large-ratio 1
docker compose start master1
```

While `master1` is down, successful responses should come from the `_2_*` pool only. After restart, traffic should return to a mix.

---

## Important Ports

| Service | Host port | Internal port |
|---|---|---|
| nginx | `8008` | `8008` |
| master1 | `7001` | `7000` |
| master2 | `7002` | `7000` |
| cpu_worker_1_1 | `9001` | `9001` |
| cpu_worker_2_1 | `9002` | `9001` |
| modal_worker_1_2 | `8000` | `8000` |
| modal_worker_2_2 | `8001` | `8000` |

---

## LLM Models Setup

### CPU Workers

CPU workers load Qwen 2.5 0.5B from `models/qwen2.5-0.5b/`. Download it once before the first build:

```powershell
python models/download_qwen.py
```

The directory is mounted read-only into each CPU worker via:

```text
./models:/app/models:ro
```

### Modal GPU Backend

The Docker modal workers do not load the large model locally. They forward requests to the deployed Modal endpoint instead.

For Modal deployment, the GPU-backed class in `workers/worker_modal.py` loads:

```text
Qwen/Qwen2.5-Coder-7B-Instruct
```

The Modal deployment also expects a Modal secret named:

```text
huggingface-secret
```

containing:

```text
HUGGINGFACEHUB_API_TOKEN
```

---

## GPU Worker Nodes (Modal)

Each local modal worker is a FastAPI shim that exposes `/generate` and `/health`, sends heartbeats to its owning master, and forwards inference to the deployed Modal endpoint.

### Architecture

```text
master -> http://modal_worker_X_2:8000/generate -> deployed Modal endpoint -> QwenCoderModel on A10G
```

### Heartbeats

`worker_modal.py` sends a heartbeat every 5 seconds to `MASTER_URL/heartbeat`:

```json
{
  "worker_id": "modal_worker_1_2",
  "url": "http://modal_worker_1_2:8000",
  "device_type": "gpu",
  "active_requests": 0,
  "queue_depth": 0,
  "timestamp": 1714838400.0
}
```

The master treats workers as stale after `WORKER_STALE_SEC` seconds without a heartbeat.

### API

- `GET /health` - liveness, returns worker metadata
- `POST /generate` - forwards inference

Request:

```json
{ "prompt": "What is distributed computing?", "max_new_tokens": 256 }
```

Response:

```json
{
  "response": "Distributed computing refers to...",
  "modal_task_id": "...",
  "modal_endpoint_task_id": "...",
  "modal_region": "...",
  "modal_model": "Qwen/Qwen2.5-Coder-7B-Instruct"
}
```

### Env Vars Used by Modal Workers

| Variable | Purpose |
|---|---|
| `MODAL_WORKER_URL` | URL of the deployed Modal endpoint |
| `MODAL_API_KEY` | Optional bearer token for the Modal endpoint |
| `WORKER_ID` | Heartbeat identity |
| `SELF_URL` | URL the master will call |
| `MASTER_URL` | Heartbeat target |
| `WORKER_API_KEY` | Optional `X-API-Key` header on heartbeats and worker calls |

---

## Worker API Contract

The worker types speak slightly different payload shapes. The master's forwarder builds the right request per `device_type`.

### CPU worker (`worker_router.py`)

```text
POST /generate
Request:  {"question": "...", "max_new_tokens": 256}
Response: {"question": "...", "answer": "..."}
```

### Modal worker (`worker_modal.py`)

```text
POST /generate
Request:  {"prompt": "...", "max_new_tokens": 256}
Response: {"response": "...", ...modal metadata...}
```

### Master `/generate` (client-facing)

```text
POST /generate
Request:  {"prompt": "...", "max_new_tokens": 256}
Response: {"response": "...", "worker_id": "...", "master_id": "...", ...}
```

The master accepts a unified `prompt` field from clients. On output it normalizes by reading `answer` from CPU workers or `response` from modal workers into the client-facing `response` field.

---

## Notes

- `nginx/nginx.conf` is mounted into the container, so config changes do not require an image rebuild.
- Master and worker images do require rebuilds: `docker compose up --build`.
- `.env` must never be committed.
- The Kaggle setup in `workers/KAGGLE_SETUP.md` is retired.
- The old Groq-based topology is no longer the active path for the current local stack.

---

## Current Scope

- NGINX least-conn load balancing across 2 master nodes
- Master scheduler with threshold-based GPU/CPU routing plus JSQ(2) on CPU
- Worker registry with heartbeat-based liveness and failure cooldowns
- Forwarder with retries on transient failures
- CPU workers (local Qwen 2.5 0.5B) plus Modal-backed GPU workers
- End-to-end test scripts (`simulate_nginx_lb.py`, `simulate_worker_scheduling.py`, `smoke_worker_node.py`)
- Modal task-id observability in the GPU response path
- RAG integration and broader fault-tolerance demos remain future work
