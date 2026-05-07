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
  ┌─────────────────────┬───────────────────────────┐
  │                     │                           │
CPU workers       Thunder GPU worker
(local docker)    (remote rented GPU instance,
                   Llama 3.1 8B Instruct)
```

Two master nodes load-balanced by nginx. Each master runs a scheduler that picks workers based on request size (large requests → GPU only; small requests → GPU first with overflow to CPU). Workers heartbeat every 5s; failed workers go into exponential cooldown. The Thunder GPU worker connects from a rented instance over a `tnr connect -t` port forward + a `cloudflared` reverse tunnel.

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

### 4. Thunder GPU worker (optional, for the GPU layer)

The full Thunder Compute setup — provisioning the instance, uploading code with `tnr scp`, opening the `cloudflared` tunnel, running `tnr connect 0 -t 8000 -t 8001` for port forwarding, and starting the worker in tmux — is documented in detail in [`onboard.md`](onboard.md#thunder-compute-gpu-worker--full-setup). A condensed run guide is in [`workers/THUNDER_SETUP.md`](workers/THUNDER_SETUP.md).

In short: Thunder hosts the FastAPI worker process on a rented GPU. A `cloudflared` quick tunnel exposes your master so the worker can heartbeat home; a `tnr connect -t` port-forward lets the master reach the worker for `/generate` calls. No extra container is built locally — the Thunder worker is a Python process on the rented box.

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
- **tnr CLI** — Thunder Compute SSH wrapper with built-in port forwarding (`tnr connect -t`)
- **tmux** — keeping worker processes alive across SSH sessions
- (planned) **ChromaDB + sentence-transformers** for RAG, **Prometheus** for metrics

---

## Documentation

- [`onboard.md`](onboard.md) — full architecture, Thunder Compute walkthrough, API contract, verification, ports
- [`workers/THUNDER_SETUP.md`](workers/THUNDER_SETUP.md) — short Thunder run guide (env vars, troubleshooting)
- [`workers/KAGGLE_SETUP.md`](workers/KAGGLE_SETUP.md) — retired, kept for historical reference
