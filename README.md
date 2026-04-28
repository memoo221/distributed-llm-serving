# CSE354 — Distributed LLM Serving System
**Ain Shams University | Faculty of Engineering | 2nd Semester 2025/2026**


## Project Structure

```
cse354_project/
├── common/        # Shared models (Request, Response, WorkerStatus) and config
├── lb/            # Load Balancer — 3 strategies + FastAPI service (port 8000)
├── master/        # Master Scheduler + metrics endpoint (port 7000)
├── workers/       # GPU Worker nodes (ports 9001–9004)
├── llm/           # PyTorch LLM inference nestam3l huggingface 
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
```

---



---

## Technologies Used

- **Python 3.10+**
- **PyTorch + HuggingFace Transformers** — LLM inference
- **FastAPI + uvicorn** — distributed services
- **ChromaDB + sentence-transformers** — RAG pipeline
- **httpx** — async HTTP communication between services
- **Prometheus** — metrics and monitoring
