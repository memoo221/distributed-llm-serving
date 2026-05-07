# Distributed LLM Inference — Onboarding

## Overview

This project implements a distributed system for handling concurrent LLM inference requests with load balancing, scheduler-driven worker selection, heartbeat-based health monitoring, and fault tolerance.

Current request flow:

```
Client / Test UI → NGINX:8008 → Master Nodes:7000 → Worker Nodes (local CPU + remote Thunder GPU)
```

Workers are heterogeneous:

- **CPU workers** load Qwen 2.5 0.5B locally (in-process slot, single-stream).
- **Thunder GPU workers** run a CUDA-backed FastAPI server on a rented [Thunder Compute](https://thundercompute.com) instance, hosting a real GPU model (default: Llama 3.1 8B Instruct). They count as `device_type: "gpu"` for scheduling.

The master's scheduler does threshold-based routing: requests with an estimated token score ≥ 256 go GPU-only; smaller requests prefer GPU while it has headroom, with overflow to CPU via JSQ(2) ([master/services/scheduler.py](master/services/scheduler.py)).

The **client test UI** at `http://localhost:8050` lets you start/stop services, view live worker registry across all masters, run load tests with configurable concurrency, view per-request responses, and export results as CSV.

> **About Groq**: a previous version of this project used hosted Groq inference (`llama-3.1-8b-instant`) as the GPU layer. That code is still present (`workers/groq_worker.py`, `workers/Dockerfile.groq`, dormant compose entries) but is **not started by default**. Thunder replaced Groq because rate limits on the free tier capped throughput at ~3 req/s.

---

## Stack

The local Docker Compose stack runs the masters, NGINX, and CPU workers. **Thunder GPU workers run outside docker** on a rented Thunder Compute instance and connect over a tunnel — see the Thunder section below.

### What's included (running locally)

- NGINX as the public entry point on port `8008` (least-conn LB across masters)
- 2 master nodes: `master1`, `master2` (FastAPI on internal port `7000`; mapped to host `7001`/`7002`)
- 2 CPU workers: `cpu_worker_1_1`, `cpu_worker_2_1` (one per master, port `9001` internal)
- (dormant) `groq_worker_*` services in compose, kept for revertability — not started

### What runs remotely

- 1 Thunder GPU worker (asymmetric setup with Llama 3.1 8B, registered with master1)

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

Expected `Up`: `distributed-nginx`, `master1`, `master2`, `cpu_worker_1_1`, `cpu_worker_2_1`. The Thunder worker is launched separately on the Thunder instance.

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

## Thunder Compute GPU Worker — Full Setup

The Thunder worker is a Python FastAPI process running on a rented GPU instance. It heartbeats back to a master and serves `/generate` calls.

### Architecture

```
Your laptop (NAT'd)                       Thunder GPU instance (public-ish IP)
┌─────────────────────────────────────┐   ┌────────────────────────────────────┐
│ docker:                             │   │ uvicorn :8000                      │
│   nginx, master1, master2, CPU      │   │   thunder_worker.py                │
│ host processes:                     │   │   loads Llama 3.1 8B on CUDA       │
│   client UI :8050                   │   │                                    │
│   2× cloudflared (master tunnels)   │◄──┤ heartbeat → master cloudflared URL │
│   1× tnr connect -t 8000 -t 8001    │──►│ /generate ← master via host.docker │
│      (port forwards laptop→Thunder) │   │                                    │
└─────────────────────────────────────┘   └────────────────────────────────────┘
```

Why this shape:

- **Worker → master** (heartbeats): master is behind your laptop's NAT. We expose master1 (and master2) over a public URL using `cloudflared`'s anonymous "Quick Tunnel" feature, so the Thunder worker has somewhere reachable to POST heartbeats.
- **Master → worker** (`/generate` calls): Thunder's instance firewalls inbound ports by default. Instead of opening them, we use `tnr connect -t PORT` to forward the laptop's localhost ports to the Thunder instance. The master container reaches the worker via `http://host.docker.internal:8000`, which Docker Desktop maps to the Windows host's localhost, which the `tnr -t` forward delivers to Thunder.

### Step 1 — Provision a Thunder instance

1. Sign up at [thundercompute.com](https://thundercompute.com) and install the `tnr` CLI.
2. Authenticate: `tnr login`
3. Create an instance with enough VRAM for your model:
   - Llama 3.1 8B in fp16 needs ~16GB — pick a GPU with ≥ 20GB to be safe (A100, A10, etc.)
   - Smaller models (Qwen 0.5B) fit on T4s (16GB)
4. Find your instance ID:
   ```powershell
   tnr list
   ```
   The instance ID is typically `0` for your first one.

### Step 2 — Upload the worker code to Thunder

From the project root on your laptop, upload just the two files the Thunder worker needs:

```powershell
tnr scp workers/thunder_worker.py workers/thunder_requirements.txt 0:
```

Both files land in `/home/ubuntu/` on instance 0. The worker is self-contained — it doesn't import anything from the rest of the project, so there's no need to ship the whole repo.

> If you change `thunder_worker.py` later, re-run the same `tnr scp` command to push the new version (then restart the worker process on Thunder).

### Step 3 — Install Python dependencies on Thunder

SSH into the instance and install. **This is the only time you use a plain `tnr connect 0` (no `-t` flag).**

```powershell
tnr connect 0
```

Then on Thunder (the two files are already in `~/` from step 2):

```bash
cd ~
pip install -r thunder_requirements.txt
exit           # back to your laptop
```

### Step 4 — Start the cloudflared tunnel for master1 (heartbeat path)

The Thunder worker needs a public URL to POST heartbeats. We use `cloudflared`'s no-account quick tunnel.

```powershell
# Install once, if you haven't:
winget install --id Cloudflare.cloudflared

# Start the tunnel for master1 — leave this terminal OPEN
cloudflared tunnel --url http://localhost:7001
```

Cloudflared prints a URL like:

```
Your quick Tunnel has been created! Visit it at:
https://wandering-mountain-1234.trycloudflare.com
```

Copy this URL. **Do not close this terminal** — closing it kills the tunnel.

> **Optional second tunnel for master2:** if you ever run a *second* Thunder worker registered to master2, repeat this step in another terminal with port `7002`. With the current asymmetric Llama 3.1 8B setup (one Thunder worker only), one tunnel is enough.

### Step 5 — Open the SSH port forward (`tnr connect -t`)

In a **separate terminal** on your laptop, run `tnr connect` with port-forwarding flags. This drops you into a shell on Thunder *and* forwards `localhost:8000` (and optionally `:8001`) on your laptop to the same ports on Thunder:

```powershell
tnr connect 0 -t 8000 -t 8001
```

You're now SSH'd into Thunder. **Do not exit this shell** — the port forward dies if you exit. Use this same shell (with `tmux`) to run the worker process in step 6.

> Why both `-t 8000` and `-t 8001`? The original symmetric setup (Qwen 0.5B, two workers) used both ports — one per master. The asymmetric Llama setup only needs `-t 8000`, but specifying both is harmless.

### Step 6 — Run the worker in tmux (inside the `tnr connect -t` shell)

The `tnr connect -t` shell must stay alive. To run multiple persistent commands in it, use `tmux`:

```bash
tmux new -s workers
```

In the tmux pane, start the Thunder worker. Set `HF_TOKEN` because Llama 3.1 is gated:

```bash
export HF_TOKEN=hf_yourtoken_here

PYTHONUNBUFFERED=1 \
MASTER_URL=https://wandering-mountain-1234.trycloudflare.com \
SELF_URL=http://host.docker.internal:8000 \
WORKER_ID=thunder_worker_1 \
uvicorn thunder_worker:app --host 0.0.0.0 --port 8000
```

Replace the cloudflared URL with the one you copied in step 4. You should see:

```
[thunder_worker_1] loading meta-llama/Llama-3.1-8B-Instruct on cuda...
... (model downloads from HF, ~16GB, takes a few minutes the first time)
[thunder_worker_1] loaded in 90.0s, vram_used=15800MB
[thunder_worker_1] master=https://wandering-mountain-1234.trycloudflare.com, self_url=http://host.docker.internal:8000
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Detach from tmux with `Ctrl+B` then `D` so it keeps running. Reattach later with `tmux attach -t workers`.

### Step 7 — Verify

On your laptop:

```powershell
# Master1 should now show the Thunder worker
curl.exe http://localhost:7001/scheduler/workers
```

Look for `thunder_worker_1` with `device_type: "gpu"` and `last_seen_sec_ago < 10`.

End-to-end:

```powershell
Invoke-RestMethod -Uri "http://localhost:8008/generate" -Method Post -ContentType "application/json" -Body '{"prompt":"Say hello","max_new_tokens":32}'
```

The `worker_id` in the response should sometimes be `thunder_worker_1` (when nginx routes to master1 and the GPU is picked).

Or open the test UI at `http://localhost:8050` — Thunder workers appear in the Live Workers panel.

### Terminals you must keep open

For Thunder to keep working, the following terminals/shells **must stay running** on your laptop:

| Terminal | What it runs | Closes if you... |
|---|---|---|
| 1 | `cloudflared tunnel --url http://localhost:7001` (master1 tunnel) | close terminal |
| 2 | `tnr connect 0 -t 8000 -t 8001` (port forwards + Thunder shell with tmux) | exit the shell |
| 3 | `python -m client.app` (test UI — optional) | close terminal |

Inside terminal 2, the worker process runs in a tmux pane. tmux survives if SSH drops *as long as the SSH connection itself stays open* — but if `tnr connect` exits, the tunnels die and Thunder can no longer be reached, even though the worker process is still running on the instance.

### Recovering from a disconnect

If `tnr connect` drops or you accidentally close terminal 2:

```powershell
tnr connect 0 -t 8000 -t 8001
tmux attach -t workers   # the worker is still running, you just need to re-tunnel
```

If `cloudflared` drops:

```powershell
cloudflared tunnel --url http://localhost:7001
```

Then **update the worker's `MASTER_URL`** because the trycloudflare URL is random and changes on every restart:

```bash
# In tmux, kill the worker and restart it with the new URL
# (Ctrl+C to kill, then re-run with new MASTER_URL)
```

If this URL churn becomes annoying, sign up for a free Cloudflare account, claim a free domain from Cloudflare Registrar, and use a **named tunnel** for stable URLs.

### Configuration

Env vars on the Thunder worker:

| Variable | Purpose | Default |
|---|---|---|
| `MODEL_NAME` | HF Hub id or local path | `meta-llama/Llama-3.1-8B-Instruct` |
| `WORKER_ID` | Heartbeat identity | `thunder_worker_1` |
| `SELF_URL` | URL the master uses to call back | `http://localhost:8000` |
| `MASTER_URL` | Heartbeat target (cloudflared URL) | (empty — heartbeats skipped) |
| `HEARTBEAT_INTERVAL` | seconds between heartbeats | `5` |
| `HF_TOKEN` | required for gated models like Llama | n/a |
| `PYTHONUNBUFFERED` | flush stdout (recommended for live logs) | `1` |

### Asymmetric setup details

With Llama 3.1 8B (~16GB VRAM in fp16), only one Thunder worker fits per GPU. The current setup is:

- master1 → has both `cpu_worker_1_1` (CPU) AND `thunder_worker_1` (GPU)
- master2 → has only `cpu_worker_2_1` (CPU)

NGINX still load-balances incoming requests across both masters (least_conn). When a request lands on master2, it goes to CPU. When it lands on master1, it goes to GPU first, then spills to CPU when GPU saturates.

To restore the symmetric setup with two GPU workers, swap the model to a smaller one (e.g. Qwen 0.5B) and run two worker processes on Thunder, one per port — but you'll lose the model quality of Llama 3.1 8B.

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
  "worker_id": "thunder_worker_1",
  "url": "http://host.docker.internal:8000",
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

While master1 is down, every successful response should come from a `_2_*` worker (CPU only, since Thunder is registered with master1). After restart, traffic returns to a mix.

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
