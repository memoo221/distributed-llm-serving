# Distributed LLM Inference — Onboarding

## Overview

This project implements a distributed system for handling concurrent LLM inference requests with load balancing, scheduler-driven worker selection, heartbeat-based health monitoring, and fault tolerance.

Request flow (two entrypoints):

```
# Direct inference (no retrieval)
Client → NGINX:8008 → Master Nodes:7000 → Worker Nodes (local CPU + Groq cloud)

# RAG inference (retrieval + prompt enhancement)
Client → NGINX:8008 (/rag/*) → RAG Service → Master Node (selected by NGINX) → Worker Nodes
```

In the RAG path, NGINX selects a `master_id` per request and forwards it to the RAG service via the `X-Master-Id` header. The RAG service then forwards the enhanced prompt to that specific master.

Workers are heterogeneous:

- **CPU workers** load Qwen 2.5 0.5B locally (in-process slot, single-stream).
- **Thunder GPU workers** run a CUDA-backed FastAPI server on rented [Thunder Compute](https://thundercompute.com) instances, hosting a real GPU model (default: Llama 3.1 8B Instruct). They count as `device_type: "gpu"` for scheduling.

The default cluster shape is **symmetric**: 2 Thunder instances × 1 worker each on `cuda:0` = **2 GPU workers total**, one heartbeating to `master1` and one to `master2`. CPU workers stay 1-per-master. Each instance has a second A100 idle on `cuda:1` — running a second worker per instance was tried and consistently caused CUDA stalls on the second process; see [`workers/THUNDER_SETUP.md`](workers/THUNDER_SETUP.md) for the rationale.

The master's scheduler does threshold-based routing: requests with an estimated token score ≥ 256 go GPU-only; smaller requests prefer GPU while it has headroom, with overflow to CPU via JSQ(2) ([master/services/scheduler.py](master/services/scheduler.py)).

The **client test UI** at `http://localhost:8050` lets you start/stop services, view live worker registry across all masters, run load tests with configurable concurrency, view per-request responses, and export results as CSV.

> **About Groq**: a previous version of this project used hosted Groq inference (`llama-3.1-8b-instant`) as the GPU layer. That code is still present (`workers/groq_worker.py`, `workers/Dockerfile.groq`, dormant compose entries) but is **not started by default**. Thunder replaced Groq because rate limits on the free tier capped throughput at ~3 req/s.

---

## Stack

The local Docker Compose stack runs the masters, NGINX, and CPU workers. **Thunder GPU workers run outside docker** on rented Thunder Compute instances and reach the masters via cloudflared tunnels (heartbeats) and `tnr ports forward` HTTPS URLs (request callbacks) — see the Thunder section below.

### What's included (running locally)

- NGINX as the public entry point on port `8008`
  - `/` routes to masters directly (least-conn LB)
  - `/rag/` routes to the RAG service (and injects `X-Master-Id`)
- 2 master nodes: `master1`, `master2` (FastAPI on internal port `7000`; mapped to host `7001`/`7002`)
- Qdrant vector DB: `qdrant` (port `6333`, persisted on a named volume)
- RAG service: `rag` (internal port `8090`, normally accessed through NGINX)
- 2 CPU workers: `cpu_worker_1_1`, `cpu_worker_2_1` (one per master, port `9001` internal)
- (dormant) `groq_worker_*` services in compose, kept for revertability — not started

### What runs remotely

- 2 Thunder GPU workers (Llama 3.1 8B) — one per instance on `cuda:0`. Instance 0's worker registers with master1; instance 1's with master2. Continuous batching with `BATCH_SIZE=64` (advertised as `slots` in heartbeats). See [`workers/THUNDER_SETUP.md`](workers/THUNDER_SETUP.md) for the walkthrough.

### Files

- `docker-compose.yml` (Groq services gated behind `profiles: ["groq"]` — not started by default)
- `nginx/nginx.conf`
- `master/Dockerfile.master_node`, `master/master_app.py`, `master/services/*` (registry, forwarder, scheduler), `master/routers/master_router.py`
- `workers/Dockerfile.cpu`, `workers/worker_router.py`, `workers/worker_service.py` — local CPU pipeline
- `workers/thunder_worker.py`, `workers/thunder_requirements.txt`, `workers/launch_workers.sh`, `workers/THUNDER_SETUP.md` — remote GPU worker + launch script for Thunder Compute
- `workers/Dockerfile.groq`, `workers/groq_worker.py` — dormant Groq cloud shim (gated behind `--profile groq`)
- `client/app.py`, `client/runner.py`, `client/docker_control.py`, `client/static/index.html` — test UI
- `scripts/redeploy.ps1` — one-shot Thunder worker redeploy (parallel scp + launch + heartbeat verify)
- `tests/simulate_nginx_lb.py` — nginx-level master LB test
- `tests/simulate_worker_scheduling.py` — end-to-end scheduler test
- `tests/smoke_worker_node.py` — direct smoke test of `WorkerNode` (CPU + remote)

### Required environment variables

Create a `.env` file (not committed) and fill values referenced by `docker-compose.yml`:

```
GROQ_API_KEY_1=          # required for groq workers — see https://console.groq.com/keys
GROQ_API_KEY_2=
GROQ_API_KEY_4=
GROQ_API_KEY_5=

# Optional: header used on master ↔ worker calls
WORKER_API_KEY=
```

RAG + Qdrant are configured via compose env vars (can be overridden):

- `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_COLLECTION`
- `EMBEDDING_MODEL` (default `all-MiniLM-L6-v2`)
- `MASTER_URLS` (JSON mapping allow-list of masters the RAG service may forward to)

### How to run

```powershell
docker compose up -d --build nginx master1 master2 cpu_worker_1_1 cpu_worker_2_1
docker compose ps -a
```

Build tips (much faster iteration):

- If you only changed one service, rebuild just that service:

```powershell
docker compose up -d --build rag
```

- Build caching: the Dockerfiles use BuildKit cache mounts for pip downloads. If caching seems disabled, set:

```powershell
$env:DOCKER_BUILDKIT=1
```

Expected `Up`: `distributed-nginx`, `qdrant`, `rag`, `master1`, `master2`, `cpu_worker_1_1`, `cpu_worker_2_1`, and thunder workers are launched separately on the Thunder instances.

### How to verify (local stack only)

**1. NGINX health:**

```powershell
curl http://localhost:8008/nginx/health
```

**1b. RAG health (through NGINX):**

```powershell
curl http://localhost:8008/rag/health
```

**1c. RAG Swagger (through NGINX):**

Open:

- http://localhost:8008/rag/docs

In Swagger you will see RAG endpoints like `POST /documents/pdf` and `POST /generate`.

**2. Each master sees its 2 workers:**

```powershell
curl.exe http://localhost:7001/scheduler/workers
curl.exe http://localhost:7002/scheduler/workers
```

`last_seen_sec_ago` should be < 5 for each CPU worker. Thunder workers will only appear here once they start and heartbeat in (Thunder section below).

**3. End-to-end via NGINX (CPU only, before Thunder is up):**

```powershell
Invoke-RestMethod -Uri "http://localhost:8008/generate" -Method Post -ContentType "application/json" -Body '{"prompt":"hello","max_new_tokens":32}'
```

Response includes `master_id` and `worker_id` so you can see which path served the request.

**4. RAG end-to-end via NGINX:**

```powershell
curl -X POST http://localhost:8008/rag/generate -H "Content-Type: application/json" -d '{"prompt":"Explain consistent hashing.","max_new_tokens":64,"top_k":3,"book_id":2}'
```

**4b. Ingest a PDF into RAG (stores chunks in Qdrant):**

```powershell
curl -X POST http://localhost:8008/rag/documents/pdf `
  -F "file=@./docs/my.pdf" `
  -F "book_id=2" `
  -F "chunk_size=1200" `
  -F "chunk_overlap=200" `
  -F "min_chunk_chars=50"
```

Notes:

- For large PDFs, you may need to increase `client_max_body_size` in `nginx/nginx.conf`.
- The payload stored per chunk includes `book_id`, `page_number`, `chunk_index`, `text`, `filename`.

Notes:

- `top_k` controls how many chunks are retrieved.
- `book_id` is used as a Qdrant payload filter (`{"book_id": <id>}`) if present.
- NGINX injects `X-Master-Id`, so the request is pinned to a specific master.

### Test scripts

```powershell
# Verify nginx least_conn distributes across masters
python tests/simulate_nginx_lb.py --base-url http://localhost:8008 --requests 30 --concurrency 8

# Verify scheduler routes by token score (small→CPU, large→GPU)
python tests/simulate_worker_scheduling.py --base-url http://localhost:8008 --requests 20 --large-ratio 1
python tests/simulate_worker_scheduling.py --base-url http://localhost:8008 --requests 20 --large-ratio 0
python tests/simulate_worker_scheduling.py --base-url http://localhost:8008 --requests 30 --large-ratio 0.3 --concurrency 8
```

Expected distribution by worker:

| Run | Workers you should see |
|---|---|
| `--large-ratio 1` | one or more of `groq_worker_1_2`, `groq_worker_1_3`, `groq_worker_2_2`, `groq_worker_2_3` |
| `--large-ratio 0` | `worker_1_1` + `worker_2_1` |
| `--large-ratio 0.3` | mix of all four |

### Failover check

Stop a master and confirm nginx fails over to the survivor:

```powershell
docker compose stop master1
python tests/simulate_worker_scheduling.py --base-url http://localhost:8008 --requests 10 --large-ratio 1
docker compose start master1
```

While master1 is down, every successful response should come from a `_2_*` worker. After restart, traffic returns to a mix.

### Important ports

| Service | Host port | Internal port |
|---|---|---|
| nginx | `8008` | 8008 |
| master1 | `7001` | 7000 |
| master2 | `7002` | 7000 |
| rag | `—` (via nginx) | 8090 |
| qdrant | `6333` | 6333 |
| cpu_worker_1_1 | `9001` | 9001 |
| cpu_worker_2_1 | `9003` | 9001 |
| client UI | `8050` | n/a (host process) |

---

## Test UI / Client

A web UI for running load tests and managing services lives at `client/`. It runs on the **host** (your laptop), not in docker, so it can shell out to `docker compose`.

```powershell
pip install fastapi uvicorn httpx pydantic
python -m client.app
```

Open `http://localhost:8050`. The UI has three panels:

- **Nodes** — auto-populates from `docker-compose.yml`, lets you Start/Stop individual services or Start all / Stop all.
- **Live Workers** — pulls from each master's `/scheduler/workers` every 3 seconds. Surfaces remote workers (Thunder) that aren't in docker-compose.
- **Run Load Test** — configurable target URL, requests, concurrency, prompt, max_new_tokens, request timeout. Live stats (throughput, p50/p95/p99 latency, per-worker breakdown), live tail of recent requests, click-to-expand response text, Export CSV.

---

## Thunder Compute GPU Workers

The full Thunder walkthrough (provisioning, key permissions, code upload,
`tnr ports forward`, cloudflared tunnels, the multi-GPU `WORKER_DEVICE`
patch, and troubleshooting) lives in [`workers/THUNDER_SETUP.md`](workers/THUNDER_SETUP.md).

### What it looks like at runtime

The default cluster shape: 2 Thunder instances, one worker each on
`cuda:0`, each registered to a different master:

```
Your laptop (NAT'd)                                  Thunder instance A (id 0, uuid q5lbf7oe)
┌──────────────────────────────────────────┐        ┌─────────────────────────────────────────┐
│ docker: nginx, master1, master2, CPUs    │        │ uvicorn :8000  thunder_a_gpu0 → cuda:0  │
│ cloudflared :7001 → MASTER1_URL          │ ◄──────┤   heartbeats to MASTER1_URL             │
│ cloudflared :7002 → MASTER2_URL          │        │   (cuda:1 left idle — see "Why one      │
│                                          │        │    worker per instance" below)          │
│ master /generate → https://<uuid>-8000.. │ ──────►│   exposed via `tnr ports forward`       │
└──────────────────────────────────────────┘        └─────────────────────────────────────────┘
                                                     Thunder instance B (id 1, uuid tn9t67pp)
                                                     ┌─────────────────────────────────────────┐
                                                     │ uvicorn :8000  thunder_b_gpu0 → cuda:0  │
                                                     │   heartbeats to MASTER2_URL             │
                                                     └─────────────────────────────────────────┘
```

Why this shape:

- **Worker → master** (heartbeats): masters live behind your laptop's
  NAT. Each master is exposed via a `cloudflared` Quick Tunnel so workers
  have a public URL to POST heartbeats to.
- **Master → worker** (`/generate` calls): `tnr ports forward` exposes
  Thunder ports at public HTTPS URLs (`https://<uuid>-<port>.thundercompute.net`),
  which masters call directly. No laptop relay needed.
- **GPU pinning**: Thunder's CUDA runtime intercepts `CUDA_VISIBLE_DEVICES`
  and routes everything to the first physical GPU. The worker honors a
  `WORKER_DEVICE` env var (`cuda:0` / `cuda:1`) and calls
  `model.to(WORKER_DEVICE)` explicitly, which PyTorch respects.
- **Continuous batching**: each worker coalesces up to `BATCH_SIZE`
  concurrent `/generate` calls (default 64) into a single
  `model.generate()` per `BATCH_WAIT_MS` window (default 30 ms). The
  worker advertises `slots: BATCH_SIZE` in heartbeats so the master
  dispatches that many concurrent requests per worker. Master's
  [`WorkerState.slots`](master/services/models.py) reads `raw["slots"]`
  from the heartbeat and falls back to global `GPU_SLOTS` only when
  unset — that's how the per-worker override works.
- **Why one worker per instance** (not two): we tried running 2 workers
  per Thunder instance (one on each A100 via `cuda:0` + `cuda:1`).
  Thunder's "Prototyping" mode CUDA virtualization consistently stalled
  the second process — direct curl to a hung worker confirms its
  `model.generate()` never returns. Until Thunder fixes the underlying
  issue (or you switch to "Production" mode / different provider), one
  worker per instance is the only reliable shape.

### Worker → master mapping

| Worker ID | Instance | GPU | Port | SELF_URL | Heartbeats to |
|---|---|---|---|---|---|
| `thunder_a_gpu0` | 0 (`q5lbf7oe`) | 0 | 8000 | `https://q5lbf7oe-8000.thundercompute.net` | master1 |
| `thunder_b_gpu0` | 1 (`tn9t67pp`) | 0 | 8000 | `https://tn9t67pp-8000.thundercompute.net` | master2 |

Each master ends up with 1 CPU + 1 GPU worker. NGINX least-conn distributes
requests roughly 50/50 across masters.

### Persistent terminals

| Terminal | What it runs | Closes if you... |
|---|---|---|
| A | `cloudflared tunnel --url http://localhost:7001` (master1) | close terminal |
| B | `cloudflared tunnel --url http://localhost:7002` (master2) | close terminal |
| (host) | `python -m client.app` (test UI — optional) | close terminal |

Thunder side requires no persistent local terminal: `tnr ports forward` is
server-side persistent and tmux sessions on Thunder survive SSH disconnects.

### Redeploying workers

The full pipeline (push code, kill old, launch new, wait for heartbeats) is
automated by [`scripts/redeploy.ps1`](scripts/redeploy.ps1):

```powershell
# Idempotent: skips if all workers are healthy
.\scripts\redeploy.ps1

# Force redeploy (after editing thunder_worker.py, changing BATCH_SIZE, etc.)
.\scripts\redeploy.ps1 -Force
```

Edit the `$Config` block at the top to change cloudflared URLs, BATCH_SIZE,
or HF token. The script runs both instances in parallel via `Start-Job` and
polls master registries until all workers heartbeat (or fails after 6 min).

### Verify after launch

```powershell
curl.exe http://localhost:7001/scheduler/workers   # CPU + thunder_a_gpu0
curl.exe http://localhost:7002/scheduler/workers   # CPU + thunder_b_gpu0

Invoke-RestMethod -Uri "http://localhost:8008/generate" -Method Post `
  -ContentType "application/json" `
  -Body '{"prompt":"hello","max_new_tokens":32}'
```

For the full step-by-step (CLI install, key permissions fix, model
gating workarounds, etc.), follow [`workers/THUNDER_SETUP.md`](workers/THUNDER_SETUP.md).

---

## LLM Models Setup

The CPU workers load Qwen 2.5 0.5B from `models/qwen2.5-0.5b/`. Download once before the first build:

```powershell
python models/download_qwen.py
```

The directory is mounted read-only into each CPU worker via the `./models:/app/models:ro` volume in compose. The download script resumes automatically if interrupted.

Thunder workers download their model from HuggingFace Hub on first startup (~16GB for Llama 3.1 8B). For gated models, set `HF_TOKEN` in the worker's environment.

---

## Worker API Contract

All workers speak HTTP. The master's `Forwarder` ([master/services/forwarder.py](master/services/forwarder.py)) builds the right payload per `device_type`.

### CPU worker (`worker_router.py`)

```
POST /generate
Request:  {"question": "...", "max_new_tokens": 256}
Response: {"question": "...", "answer": "..."}
```

### Thunder GPU worker (`thunder_worker.py`)

```
POST /generate
Request:  {"prompt": "...", "max_new_tokens": 256}
Response: {"response": "..."}
```

### Master `/generate` (client-facing)

```
POST /generate
Request:  {"prompt": "...", "max_new_tokens": 256}
Response: {"response": "...", "worker_id": "...", "master_id": "...", ...}
```

The master accepts the unified `prompt` field on input. On output it normalizes by reading `answer` (CPU) or `response` (GPU) into a single `response` field.

### Heartbeat payload (worker → master)

Posted to `MASTER_URL/heartbeat` every 5 seconds:

```json
{
  "worker_id": "thunder_a_gpu0",
  "url": "https://q5lbf7oe-8000.thundercompute.net",
  "device_type": "gpu",
  "active_requests": 0,
  "queue_depth": 0,
  "timestamp": 1714838400.0
}
```

The master's registry treats workers stale after `WORKER_STALE_SEC` (default 15s) of silence and skips them in `pick_worker`.

---

## Test scripts

```powershell
# Verify nginx least_conn distributes across masters
python tests/simulate_nginx_lb.py --base-url http://localhost:8008 --requests 30 --concurrency 8

# Verify scheduler routes by token score (small→GPU, large→GPU, mixed)
python tests/simulate_worker_scheduling.py --base-url http://localhost:8008 --requests 20 --large-ratio 1
python tests/simulate_worker_scheduling.py --base-url http://localhost:8008 --requests 20 --large-ratio 0
python tests/simulate_worker_scheduling.py --base-url http://localhost:8008 --requests 30 --large-ratio 0.3 --concurrency 8
```

For interactive load testing with live stats and CSV export, use the **client UI** at `http://localhost:8050`.

### Failover check

Stop a master and confirm nginx fails over to the survivor:

```powershell
docker compose stop master1
python tests/simulate_worker_scheduling.py --base-url http://localhost:8008 --requests 10 --large-ratio 1
docker compose start master1
```

While master1 is down, every successful response should come from master2's pool: `cpu_worker_2_1` and `thunder_b_gpu0`. After restart, traffic returns to a balanced mix across both masters.

---

## RAG API Contract

The RAG service exposes a master-compatible API, but with retrieval controls.

### NGINX → RAG

```
POST /rag/generate
Request:  {"prompt":"...","max_new_tokens":64,"top_k":3,"book_id":2}
Response: (same as master /generate)
```

`master_id` selection:

- Preferred: NGINX injects `X-Master-Id` and the RAG service forwards only to that master.
- Optional: you can also pass `master_id` in the JSON body; it must exist in the RAG allow-list (`MASTER_URLS`).

---

## Notes

- `nginx/nginx.conf` is mounted into the container — config changes don't require an image rebuild.
- Master and worker images do require rebuilds: `docker compose up -d --build master1 master2`.
- `.env` must never be committed.
- The **Kaggle integration** documented in `workers/KAGGLE_SETUP.md` is **retired**.
- The **Groq integration** (`workers/groq_worker.py`, `workers/Dockerfile.groq`) is dormant — kept in the repo for revertability but not started by default. The `groq_worker_*` services in `docker-compose.yml` are gated behind the `groq` profile. To revive: `docker compose --profile groq up -d` (and ensure `GROQ_API_KEY_1`…`_6` are set in `.env`).

---

## Current Scope

- ✅ NGINX least-conn load balancing across 2 master nodes (with `proxy_next_upstream` failover)
- ✅ Master scheduler with threshold-based GPU/CPU routing + JSQ(2) on CPU
- ✅ Worker registry with heartbeat-based liveness + exponential failure cooldowns
- ✅ Forwarder with cross-worker retries on 5xx / timeout, saturation-aware queue waiting
- ✅ CPU workers (Qwen 2.5 0.5B, local) + Thunder GPU workers (Llama 3.1 8B, remote)
- ✅ Test UI with live worker registry, configurable load tests, CSV export, response inspection
- ✅ End-to-end test scripts (`simulate_nginx_lb.py`, `simulate_worker_scheduling.py`, `smoke_worker_node.py`)
- ⏳ RAG integration (separate branch, not yet merged)
- ⏳ Real-time in-flight tracking at the master (so CPU saturation fallback fires under sub-heartbeat bursts)
- ✅ RAG service + Qdrant wired behind NGINX (`/rag/generate`)
- ⏳ Data ingestion pipeline for Qdrant (chunking + upsert) depending on your dataset
- ⏳ Multi-failure fault-tolerance demo
