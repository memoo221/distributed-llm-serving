# Distributed LLM Inference — Onboarding

## Overview

This project implements a distributed system for handling concurrent LLM inference requests with load balancing, scheduler-driven worker selection, heartbeat-based health monitoring, and fault tolerance.

Current request flow:

```
Client / Test UI → NGINX:8008 → Master Nodes:7000 → Worker Nodes (local CPU + remote Thunder GPU)
```

Workers are heterogeneous:

- **CPU workers** load Qwen 2.5 0.5B locally (in-process slot, single-stream).
- **Thunder GPU workers** run a CUDA-backed FastAPI server on rented [Thunder Compute](https://thundercompute.com) instances, hosting a real GPU model (default: Llama 3.1 8B Instruct). They count as `device_type: "gpu"` for scheduling.

The default cluster shape is **symmetric**: 2 Thunder instances × 2 A100 80GB GPUs each = **4 GPU workers total**, with 2 heartbeating to `master1` and 2 to `master2`. CPU workers stay 1-per-master.

The master's scheduler does threshold-based routing: requests with an estimated token score ≥ 256 go GPU-only; smaller requests prefer GPU while it has headroom, with overflow to CPU via JSQ(2) ([master/services/scheduler.py](master/services/scheduler.py)).

The **client test UI** at `http://localhost:8050` lets you start/stop services, view live worker registry across all masters, run load tests with configurable concurrency, view per-request responses, and export results as CSV.

> **About Groq**: a previous version of this project used hosted Groq inference (`llama-3.1-8b-instant`) as the GPU layer. That code is still present (`workers/groq_worker.py`, `workers/Dockerfile.groq`, dormant compose entries) but is **not started by default**. Thunder replaced Groq because rate limits on the free tier capped throughput at ~3 req/s.

---

## Stack

The local Docker Compose stack runs the masters, NGINX, and CPU workers. **Thunder GPU workers run outside docker** on rented Thunder Compute instances and reach the masters via cloudflared tunnels (heartbeats) and `tnr ports forward` HTTPS URLs (request callbacks) — see the Thunder section below.

### What's included (running locally)

- NGINX as the public entry point on port `8008` (least-conn LB across masters)
- 2 master nodes: `master1`, `master2` (FastAPI on internal port `7000`; mapped to host `7001`/`7002`)
- 2 CPU workers: `cpu_worker_1_1`, `cpu_worker_2_1` (one per master, port `9001` internal)
- (dormant) `groq_worker_*` services in compose, kept for revertability — not started

### What runs remotely

- 4 Thunder GPU workers across 2 instances (Llama 3.1 8B), 2 registered with master1 and 2 with master2 — see [`workers/THUNDER_SETUP.md`](workers/THUNDER_SETUP.md) for the full walkthrough

### Files

- `docker-compose.yml`
- `nginx/nginx.conf`
- `master/Dockerfile.master_node`, `master/master_app.py`, `master/services/*` (registry, forwarder, scheduler), `master/routers/master_router.py`
- `workers/Dockerfile.cpu`, `workers/worker_router.py`, `workers/worker_service.py` — local CPU pipeline
- `workers/thunder_worker.py`, `workers/thunder_requirements.txt`, `workers/THUNDER_SETUP.md` — remote GPU worker for Thunder Compute
- `workers/Dockerfile.groq`, `workers/groq_worker.py` — dormant Groq cloud shim (not started)
- `client/app.py`, `client/runner.py`, `client/docker_control.py`, `client/static/index.html` — test UI
- `tests/simulate_nginx_lb.py` — nginx-level master LB test
- `tests/simulate_worker_scheduling.py` — end-to-end scheduler test
- `tests/smoke_worker_node.py` — direct smoke test of `WorkerNode` (CPU + remote)

### Required environment variables

`.env` is required if you ever start the dormant Groq services:

```
GROQ_API_KEY=          # only if running groq workers
WORKER_API_KEY=        # optional — header used on master ↔ worker calls
```

For Thunder, env vars are set per-process on the Thunder instance (see Thunder section).

### How to run

```powershell
docker compose up -d --build nginx master1 master2 cpu_worker_1_1 cpu_worker_2_1
docker compose ps -a
```

Expected `Up`: `distributed-nginx`, `master1`, `master2`, `cpu_worker_1_1`, `cpu_worker_2_1`. Thunder workers are launched separately on the Thunder instances.

### How to verify (local stack only)

**1. NGINX health:**

```powershell
curl http://localhost:8008/nginx/health
```

**2. Each master sees its CPU worker:**

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

### Important ports

| Service | Host port | Internal port |
|---|---|---|
| nginx | `8008` | 8008 |
| master1 | `7001` | 7000 |
| master2 | `7002` | 7000 |
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

The default cluster shape uses 2 Thunder instances, each with 2× A100
80GB GPUs, running 4 worker processes total:

```
Your laptop (NAT'd)                                  Thunder instance A (id 0, uuid q5lbf7oe)
┌──────────────────────────────────────────┐        ┌─────────────────────────────────────────┐
│ docker: nginx, master1, master2, CPUs    │        │ uvicorn :8000  thunder_a_gpu0 → cuda:0  │
│ cloudflared :7001 → MASTER1_URL          │ ◄──────┤   heartbeats to MASTER1_URL             │
│ cloudflared :7002 → MASTER2_URL          │        │ uvicorn :8001  thunder_a_gpu1 → cuda:1  │
│                                          │ ◄──────┤   heartbeats to MASTER2_URL             │
│ master /generate → https://q5lbf7oe-...  │ ──────►│   exposed via `tnr ports forward`       │
└──────────────────────────────────────────┘        └─────────────────────────────────────────┘
                                                     Thunder instance B (id 1, uuid tn9t67pp)
                                                     ┌─────────────────────────────────────────┐
                                                     │ thunder_b_gpu0 → cuda:0 → MASTER1_URL   │
                                                     │ thunder_b_gpu1 → cuda:1 → MASTER2_URL   │
                                                     └─────────────────────────────────────────┘
```

Why this shape:

- **Worker → master** (heartbeats): masters live behind your laptop's
  NAT. Each master is exposed via a `cloudflared` Quick Tunnel so workers
  have a public URL to POST heartbeats to.
- **Master → worker** (`/generate` calls): `tnr ports forward` exposes
  Thunder ports at public HTTPS URLs (`https://<uuid>-<port>.thundercompute.net`),
  which masters call directly. No laptop relay needed (this replaces the
  older `tnr connect -t` flow).
- **GPU pinning**: Thunder's CUDA runtime intercepts `CUDA_VISIBLE_DEVICES`
  and routes everything to the first physical GPU. The worker honors a
  `WORKER_DEVICE` env var (`cuda:0` / `cuda:1`) and calls
  `model.to(WORKER_DEVICE)` explicitly, which PyTorch respects.

### Worker → master mapping

| Worker ID | Instance | GPU | Port | SELF_URL | Heartbeats to |
|---|---|---|---|---|---|
| `thunder_a_gpu0` | 0 (`q5lbf7oe`) | 0 | 8000 | `https://q5lbf7oe-8000.thundercompute.net` | master1 |
| `thunder_a_gpu1` | 0 (`q5lbf7oe`) | 1 | 8001 | `https://q5lbf7oe-8001.thundercompute.net` | master2 |
| `thunder_b_gpu0` | 1 (`tn9t67pp`) | 0 | 8000 | `https://tn9t67pp-8000.thundercompute.net` | master1 |
| `thunder_b_gpu1` | 1 (`tn9t67pp`) | 1 | 8001 | `https://tn9t67pp-8001.thundercompute.net` | master2 |

Master1 ends up with `cpu_worker_1_1` + 2 GPU workers; master2 with
`cpu_worker_2_1` + 2 GPU workers. NGINX least-conn distributes requests
roughly 50/50 across masters.

### Persistent terminals

| Terminal | What it runs | Closes if you... |
|---|---|---|
| A | `cloudflared tunnel --url http://localhost:7001` (master1) | close terminal |
| B | `cloudflared tunnel --url http://localhost:7002` (master2) | close terminal |
| (host) | `python -m client.app` (test UI — optional) | close terminal |

Thunder side requires no persistent local terminal: `tnr ports forward` is
server-side persistent and tmux sessions on Thunder survive SSH disconnects.

### Verify after launch

```powershell
curl.exe http://localhost:7001/scheduler/workers   # CPU + thunder_a_gpu0 + thunder_b_gpu0
curl.exe http://localhost:7002/scheduler/workers   # CPU + thunder_a_gpu1 + thunder_b_gpu1

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

While master1 is down, every successful response should come from master2's pool: `cpu_worker_2_1`, `thunder_a_gpu1`, `thunder_b_gpu1`. After restart, traffic returns to a balanced mix across both masters.

---

## Notes

- `nginx/nginx.conf` is mounted into the container — config changes don't require an image rebuild.
- Master and worker images do require rebuilds: `docker compose up -d --build master1 master2`.
- `.env` must never be committed.
- The **Kaggle integration** documented in `workers/KAGGLE_SETUP.md` is **retired**.
- The **Groq integration** (`workers/groq_worker.py`, `workers/Dockerfile.groq`) is dormant — kept in the repo for revertability but not started by default. To revive it, uncomment the `groq_worker_*` entries in `docker-compose.yml`'s `WORKERS_BOOTSTRAP` and bring those services up.

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
- ⏳ Multi-failure fault-tolerance demo
