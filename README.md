# CSE354 — Distributed LLM Serving System

**Ain Shams University | Faculty of Engineering | 2nd Semester 2025/2026**

A distributed system for serving 1000+ concurrent LLM inference requests across heterogeneous worker nodes (GPU + CPU), with load balancing, dynamic batching, RAG, and fault tolerance.

## Architecture

```
Client (1000+ users)
        ↓
NGINX (port 8008)
        ↓
Master Nodes (port 7000)  ←  scheduler, worker registry, heartbeats
        ↓
Worker Nodes (local CPU + remote Kaggle GPU)
        ↓
LLM (Qwen 2.5 0.5B) + RAG (ChromaDB)
```

## Project Structure

```
cse354_project/
├── common/        # Shared schemas (Request, Response, WorkerStatus) and config
├── lb/            # NGINX load balancer config (3 strategies)
├── master/        # Master scheduler + metrics endpoint (port 7000)
├── workers/       # Worker nodes — local + Kaggle GPU
│   ├── worker.py            # Core worker (CPU or CUDA)
│   ├── inference.py         # Model loading + batched generation + GPU stats
│   ├── kaggle_worker.py     # Kaggle-specific entrypoint for the 2× T4 setup
│   ├── README.md            # Worker usage notes
│   └── KAGGLE_SETUP.md      # Step-by-step Kaggle deployment
├── llm/           # PyTorch / HuggingFace inference helpers
├── rag/           # Retrieval-Augmented Generation (ChromaDB + sentence-transformers)
├── client/        # Async load generator (simulates 1000 users)
├── monitoring/    # Metrics and Prometheus integration
├── tests/         # Unit tests + result plots
├── logs/          # Service logs (auto-generated at runtime)
└── docs/          # Architecture diagrams and report assets
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cse354_project.git
cd cse354_project

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy env template
cp .env.example .env
# Then fill in WORKER_API_KEY (and any other required values) from the team vault
```

---

## Running

### Local CPU workers

```bash
cd workers
python worker.py cpu_worker_1 cpu 9001
python worker.py cpu_worker_2 cpu 9002
```

### Remote GPU workers (Kaggle)

The Kaggle entrypoint is `workers/kaggle_worker.py`. Full deployment walkthrough in [`workers/KAGGLE_SETUP.md`](workers/KAGGLE_SETUP.md). The notebook itself is not committed — Kaggle Secrets are used for credentials.

### Master + load balancer (Docker)

```bash
docker compose up --build -d
```

See [`onboard.md`](onboard.md) for endpoints, the worker API contract, and verification steps.

---

## Technologies Used

- **Python 3.10+**
- **PyTorch + HuggingFace Transformers** — LLM inference (Qwen 2.5 0.5B)
- **FastAPI + uvicorn** — worker and master services
- **NGINX** — Layer 7 load balancing across master nodes
- **ChromaDB + sentence-transformers** — RAG pipeline
- **httpx** — async HTTP between services
- **pynvml** — driver-level GPU telemetry
- **ngrok** — exposing Kaggle GPU workers publicly
- **Prometheus** — metrics and monitoring
- **Docker Compose** — local cluster orchestration

---

## Documentation

- [`onboard.md`](onboard.md) — full architecture, API contract, verification flow
- [`workers/README.md`](workers/README.md) — running workers locally
- [`workers/KAGGLE_SETUP.md`](workers/KAGGLE_SETUP.md) — deploying GPU workers on Kaggle
