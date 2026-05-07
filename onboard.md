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

- **CPU workers** load Qwen 2.5 0.5B locally and serve from a single in-process slot.
- **Groq workers** are thin FastAPI shims that forward to the Groq cloud API (`llama-3.1-8b-instant`). They count as `device_type: "gpu"` for scheduling.

The master's scheduler does threshold-based routing: requests with an estimated token score ≥ 256 go GPU-only; smaller requests prefer GPU while it has headroom, and overflow to CPU via JSQ(2) when GPU is busy ([master/services/scheduler.py](master/services/scheduler.py)).

---

## Stack

The local Docker Compose stack runs the full end-to-end pipeline.

### What's included

- NGINX as the public entry point on port `8008`
  - `/` routes to masters directly (least-conn LB)
  - `/rag/` routes to the RAG service (and injects `X-Master-Id`)
- 2 master nodes: `master1`, `master2` (FastAPI on internal port `7000`; mapped to host `7001`/`7002`)
- Qdrant vector DB: `qdrant` (port `6333`, persisted on a named volume)
- RAG service: `rag` (internal port `8090`, normally accessed through NGINX)
- 2 CPU workers: `cpu_worker_1_1`, `cpu_worker_2_1` (one per master, port `9001` internal)
- Groq workers (FastAPI shims): `groq_worker_1_2`, `groq_worker_1_3`, `groq_worker_2_2`, `groq_worker_2_3` (port `8000` internal)

### Files

- `docker-compose.yml`
- `nginx/nginx.conf`
- `master/Dockerfile.master_node`, `master/master_app.py`, `master/services/*` (registry, forwarder, scheduler), `master/routers/master_router.py`
- `workers/Dockerfile.cpu`, `workers/worker_router.py`, `workers/worker_service.py` — local CPU pipeline
- `workers/Dockerfile.groq`, `workers/groq_worker.py` — Groq cloud shim
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
docker compose up --build -d
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

Expected `Up`: `distributed-nginx`, `qdrant`, `rag`, `master1`, `master2`, `cpu_worker_1_1`, `cpu_worker_2_1`, and the enabled Groq workers.

### How to verify

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
curl http://localhost:7001/scheduler/workers
curl http://localhost:7002/scheduler/workers
```

`last_seen_sec_ago` should be < 5 for both workers. `device_type` should be `cpu` for one and `gpu` for the other.

**3. End-to-end via NGINX:**

```powershell
# small request (max_new_tokens < 256) → routes to a CPU worker
curl -X POST http://localhost:8008/generate -H "Content-Type: application/json" -d '{\"prompt\":\"hello\",\"max_new_tokens\":32}'

# large request (max_new_tokens ≥ 256) → routes to a Groq worker
curl -X POST http://localhost:8008/generate -H "Content-Type: application/json" -d '{\"prompt\":\"hello\",\"max_new_tokens\":300}'
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
| cpu_worker_2_1 | `9002` | 9001 |
| groq_worker_1_2 | `8000` | 8000 |
| groq_worker_1_3 | `8002` | 8000 |
| groq_worker_2_2 | `8001` | 8000 |
| groq_worker_2_3 | `8004` | 8000 |

---

## LLM Models Setup

The CPU workers load Qwen 2.5 0.5B from `models/qwen2.5-0.5b/`. Download once before the first build:

```powershell
python models/download_qwen.py
```

The directory is mounted read-only into each CPU worker via the `./models:/app/models:ro` volume in compose. The download script resumes automatically if interrupted.

> Groq workers don't need a local model — they forward to Groq cloud.

---

## GPU Worker Nodes (Groq)

Each Groq worker is a thin FastAPI service that forwards `/generate` to Groq's `chat.completions` endpoint and returns the response. From the master's perspective they are `device_type: "gpu"`.

### Architecture

```
master ──▶ http://groq_worker_*:8000/generate ──▶ Groq cloud (llama-3.1-8b-instant)
```

### Heartbeats

`groq_worker.py` runs a background `heartbeat_loop()` that posts every 5 s to `MASTER_URL/heartbeat`:

```json
{
  "worker_id": "groq_worker_1_3",
  "url": "http://groq_worker_1_3:8000",
  "device_type": "gpu",
  "active_requests": 0,
  "queue_depth": 0,
  "timestamp": 1714838400.0
}
```

The master's registry treats workers stale after `WORKER_STALE_SEC` (default 15 s) of silence and skips them in `pick_worker`.

### API

- `GET /health` — liveness, returns `worker_id` and `active_requests`
- `POST /generate` — runs inference

**Request:**

```json
{ "prompt": "What is distributed computing?", "max_new_tokens": 256 }
```

**Response:**

```json
{ "response": "Distributed computing refers to..." }
```

### Env vars (set in compose)

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY_*` | Authenticates against Groq cloud |
| `WORKER_ID` | Heartbeat identity |
| `SELF_URL` | URL the master will call back on |
| `MASTER_URL` | Heartbeat target |
| `WORKER_API_KEY` | Optional — sent as `X-API-Key` header on heartbeats |

---

## Worker API Contract

The two worker types speak slightly different request shapes; the master's `Forwarder` ([master/services/forwarder.py](master/services/forwarder.py)) builds the right payload per `device_type`.

### CPU worker (`worker_router.py`)

```
POST /generate
Request:  {"question": "...", "max_new_tokens": 256}
Response: {"question": "...", "answer": "..."}
```

### Groq worker (`groq_worker.py`)

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

The master accepts the unified `prompt` field on input. On output it normalizes by reading `answer` (CPU) or `response` (Groq) into a single `response` field.

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
- Master and worker images do require rebuilds: `docker compose up --build`.
- `.env` must never be committed.
- The Kaggle integration documented in `workers/KAGGLE_SETUP.md` is **retired**. GPU traffic now goes through Groq cloud instead of Tesla T4s on Kaggle.

---

## Current Scope

- ✅ NGINX least-conn load balancing across 2 master nodes (with `proxy_next_upstream` failover)
- ✅ Master scheduler with threshold-based GPU/CPU routing + JSQ(2) on CPU
- ✅ Worker registry with heartbeat-based liveness + failure cooldowns
- ✅ Forwarder with cross-worker retries on 5xx / timeout
- ✅ CPU workers (Qwen 2.5 0.5B, local) + Groq workers (cloud LLM)
- ✅ End-to-end test scripts (`simulate_nginx_lb.py`, `simulate_worker_scheduling.py`, `smoke_worker_node.py`)
- ⏳ Real-time in-flight tracking at the master (so CPU saturation fallback fires under sub-heartbeat bursts)
- ✅ RAG service + Qdrant wired behind NGINX (`/rag/generate`)
- ⏳ Data ingestion pipeline for Qdrant (chunking + upsert) depending on your dataset
- ⏳ Multi-failure fault-tolerance demo
