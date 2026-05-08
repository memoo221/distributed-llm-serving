# CSE354 — Distributed LLM Serving System

**Ain Shams University | Faculty of Engineering | 2nd Semester 2025/2026**

A distributed system for serving 1000+ concurrent LLM inference requests across heterogeneous worker nodes (CPU + remote GPU), with load balancing, threshold-based scheduling, heartbeat liveness, fault tolerance, and a built-in test harness.

## Architecture

```
Client (test UI or load script)
        ↓
NGINX (port 8008, least-conn)
        ↓
Master Nodes (master1 :7001, master2 :7002)
        ↓
  ┌─────────────────────┬─────────────────────────────────────┐
  │                     │                                     │
CPU workers       Thunder GPU workers
(local docker,    (4 workers across 2 rented instances,
 1 per master)     2× A100 80GB each, Llama 3.1 8B,
                   2 workers per master — symmetric)
```

Two master nodes load-balanced by nginx. Each master runs a scheduler that picks workers based on request size (large requests → GPU only; small requests → GPU first with overflow to CPU). Workers heartbeat every 5s; failed workers go into exponential cooldown. Thunder GPU workers reach the masters via `cloudflared` Quick Tunnels (heartbeats) and are reachable from masters via `tnr ports forward` HTTPS URLs (`/generate` callbacks).

## Project Structure

```
distributed-llm-serving/
├── master/         # Master scheduler, worker registry, request forwarder (port 7000)
│   ├── master_app.py
│   ├── routers/master_router.py
│   └── services/   # config, registry, scheduler, forwarder, models
├── workers/        # Worker implementations
│   ├── worker_router.py / worker_service.py    # Local CPU (Qwen 2.5 0.5B)
│   ├── thunder_worker.py                       # Remote GPU on Thunder Compute
│   ├── thunder_requirements.txt
│   ├── THUNDER_SETUP.md                        # One-page Thunder run guide
│   ├── groq_worker.py / Dockerfile.groq        # Dormant — Groq integration kept for revertability
│   ├── kaggle_worker.py / KAGGLE_SETUP.md      # Retired — old Kaggle GPU path
│   └── inference.py                            # Shared model-loading helpers
├── nginx/nginx.conf                            # Layer-7 load balancer config
├── client/         # Test UI (FastAPI + HTML)
│   ├── app.py
│   ├── runner.py
│   ├── docker_control.py
│   └── static/index.html
├── models/         # Local model files + download script
├── llm/, rag/, monitoring/, common/, lb/       # Subsystems
├── tests/          # Load + scheduler + smoke tests
├── docker-compose.yml
├── onboard.md      # ← Full setup walkthrough, including Thunder Compute
└── README.md
```

---

## Setup

### 1. Repo + Python env

```bash
git clone https://github.com/<your-fork>/distributed-llm-serving.git
cd distributed-llm-serving

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Local stack (CPU workers + masters + nginx)

```powershell
# Download the local CPU model once
python models/download_qwen.py

# Bring up the local stack
docker compose up -d --build nginx master1 master2 cpu_worker_1_1 cpu_worker_2_1
```

### 3. Test UI on the host

```powershell
pip install fastapi uvicorn httpx pydantic
python -m client.app
```

Open `http://localhost:8050`. The UI shows docker services, live worker registry across masters, and a load-test runner with CSV export.

### 4. Thunder GPU workers (optional, for the GPU layer)

The full Thunder Compute walkthrough — provisioning instances, fixing
Windows SSH key permissions, `tnr scp` of the worker code, `tnr ports
forward` for public HTTPS endpoints, two `cloudflared` Quick Tunnels for
master heartbeats, the `WORKER_DEVICE` patch for multi-GPU pinning, and
launching 4 workers in tmux — lives in [`workers/THUNDER_SETUP.md`](workers/THUNDER_SETUP.md). [`onboard.md`](onboard.md#thunder-compute-gpu-workers) has a condensed reference with the runtime architecture diagram.

In short: each Thunder instance runs N FastAPI worker processes (one per
GPU), exposed via `tnr ports forward` HTTPS URLs. `cloudflared` exposes
each master at a public URL so workers can POST heartbeats. No extra
container is built locally — Thunder workers are plain Python processes
on the rented boxes.

---

## Running

```powershell
# Start the local stack
docker compose up -d nginx master1 master2 cpu_worker_1_1 cpu_worker_2_1

# Verify
curl http://localhost:8008/nginx/health
curl.exe http://localhost:7001/scheduler/workers

# End-to-end through nginx
Invoke-RestMethod -Uri "http://localhost:8008/generate" -Method Post -ContentType "application/json" -Body '{"prompt":"hello","max_new_tokens":32}'
```

For load testing with concurrency, latency percentiles, and per-request response inspection, use the test UI at `http://localhost:8050`.

See [`onboard.md`](onboard.md) for the full verification flow, the worker API contract, port reference, and Thunder operational notes.

---

## Technologies Used

- **Python 3.10+**
- **PyTorch + HuggingFace Transformers** — LLM inference (Qwen 2.5 0.5B local, Llama 3.1 8B Instruct on Thunder)
- **FastAPI + uvicorn** — masters, CPU workers, Thunder workers, test UI
- **NGINX** — Layer-7 load balancing across master nodes (least_conn)
- **httpx** — async HTTP between masters and workers
- **Docker Compose** — local cluster orchestration (masters + nginx + CPU workers)
- **Thunder Compute** — rented GPU instances for the GPU worker layer
- **cloudflared** — public quick tunnels exposing masters to remote workers
- **tnr CLI** — Thunder Compute SSH wrapper with built-in port forwarding (`tnr ports forward` for persistent HTTPS exposure; `tnr connect -t` for ad-hoc laptop-side forwards)
- **tmux** — keeping worker processes alive across SSH sessions
- (planned) **ChromaDB + sentence-transformers** for RAG, **Prometheus** for metrics

---

## Documentation

- [`onboard.md`](onboard.md) — full architecture, Thunder Compute walkthrough, API contract, verification, ports
- [`workers/THUNDER_SETUP.md`](workers/THUNDER_SETUP.md) — full Thunder Compute walkthrough (provisioning, key permissions, `tnr ports forward`, `cloudflared`, `WORKER_DEVICE` GPU pinning, troubleshooting)
- [`workers/KAGGLE_SETUP.md`](workers/KAGGLE_SETUP.md) — retired, kept for historical reference
