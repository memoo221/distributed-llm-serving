# CSE354 — Distributed LLM Serving System

**Ain Shams University | Faculty of Engineering | 2nd Semester 2025/2026**

A distributed system for serving 1000+ concurrent LLM inference requests across heterogeneous worker nodes (CPU + remote GPU), with NGINX load balancing, threshold-based scheduling, heartbeat liveness tracking, fault tolerance with automatic retries, RAG support, and a built-in test UI.

---

## Architecture

```
Client (test UI or load script)
        ↓
NGINX :8008  (least-conn LB + proxy_next_upstream failover)
        ↓
Master Nodes (master1 :7001, master2 :7002)
        ↓
  ┌─────────────────────┬─────────────────────────────────────┐
CPU workers       Thunder GPU workers
(local docker,    (2 rented A100 80GB instances, Llama 3.1 8B,
 1 per master,     1 worker per master — symmetric.
 Qwen 2.5 0.5B)    Continuous batching, BATCH_SIZE=64)
```

Two master nodes load-balanced by NGINX. Each master runs a scheduler that picks workers based on request size and live load (large requests → GPU only; small requests → GPU first with overflow to CPU). Workers heartbeat every 5 s; failed workers enter exponential cooldown. Thunder GPU workers reach the masters via `cloudflared` Quick Tunnels (heartbeats) and are reachable from masters via `tnr ports forward` HTTPS URLs (`/generate` callbacks).

For details on the request flow, scheduling algorithm, and fault-tolerance layers, see [`onboard.md`](onboard.md).

---

## Setup

### Prerequisites (one-time, on your laptop)

```powershell
# Python 3.10+
python --version

# Docker Desktop (for the local stack)
docker --version
docker compose version

# Thunder CLI (for the GPU workers)
pip install tnr
tnr --version

# Cloudflare tunnel (for heartbeats from Thunder workers to your laptop's masters)
winget install --id Cloudflare.cloudflared
cloudflared --version

# Python deps for the test UI
pip install fastapi uvicorn httpx pydantic
```

You'll also need a **HuggingFace token** with access to `meta-llama/Llama-3.1-8B-Instruct` (or swap to an open model like `Qwen/Qwen2.5-7B-Instruct`).

### One-time: download the local CPU model

```powershell
python models/download_qwen.py
```

This pulls Qwen 2.5 0.5B once and caches it under `./models/`, mounted read-only into the CPU worker containers.

---

## Running — Full Step-by-Step

These steps must be done in order. Each opens a window that **stays open** for the duration of your session.

### Step 1 — Bring up the local Docker stack

```powershell
docker compose up -d --build nginx master1 master2 cpu_worker_1_1 cpu_worker_2_1 rag qdrant
```

Verify everything is `Up`:

```powershell
docker compose ps -a
```

You should see `distributed-nginx`, `master1`, `master2`, `cpu_worker_1_1`, `cpu_worker_2_1`, `rag`, `qdrant` all running.

Quick sanity check:

```powershell
curl http://localhost:8008/nginx/health
curl.exe http://localhost:7001/scheduler/workers
curl.exe http://localhost:7002/scheduler/workers
```

Each master should show its CPU worker with `last_seen_sec_ago < 5`.

### Step 2 — Launch the client / test UI

In a **new terminal**, run the host-side test UI:

```powershell
python -m client.app
```

Open [http://localhost:8050](http://localhost:8050) in a browser. You should see the Nodes panel (auto-populated from docker-compose) and the Live Workers panel (showing the two CPU workers from each master). Keep this terminal open.

> At this point you can already run /generate against the CPU tier through NGINX. If you only need the local CPU stack, skip to "Smoke test" below. The Thunder GPU steps (3–7) add the high-throughput tier.

### Step 3 — Start two cloudflared Quick Tunnels (one per master)

The Thunder GPU workers heartbeat *inbound* to your laptop, so each master needs a public URL. Open **two new terminals**, one for each tunnel, and leave both open for the entire session:

**Terminal A:**
```powershell
cloudflared tunnel --url http://localhost:7001
```

After ~5 seconds it prints a line like:
```
Your quick Tunnel has been created! Visit it at:
https://wherever-mixture-decimal-mart.trycloudflare.com
```
**Copy that URL — this is `MASTER1_URL`.** Do not close this terminal.

**Terminal B:**
```powershell
cloudflared tunnel --url http://localhost:7002
```
**Copy that URL — this is `MASTER2_URL`.** Do not close this terminal.

Quick verify:
```powershell
curl.exe https://<MASTER1_URL>/scheduler/workers
curl.exe https://<MASTER2_URL>/scheduler/workers
```
Each should return the same JSON as `http://localhost:7001` / `:7002`.

> The trycloudflare URLs are random and change every time you restart cloudflared. For a stable URL, set up a named tunnel with a free Cloudflare account.

### Step 4 — Provision the Thunder Compute instances

You need **two running Thunder instances** with 2× A100 80GB each. From the Thunder dashboard or CLI:

```powershell
tnr login            # opens browser, authenticate once
tnr status           # list your instances
```

Note the `ID` and `UUID` for each running instance. Example output:
```
ID  UUID      STATUS    GPU
0   gaxwh8fq  RUNNING   2× A100 80GB
1   xcidc8hb  RUNNING   2× A100 80GB
```

On a fresh Windows install you may need to lock down the Thunder SSH keys once — see [`workers/thunder/THUNDER_SETUP.md`](workers/thunder/THUNDER_SETUP.md#step-1--fix-windows-ssh-key-permissions) for the `icacls` block.

Open Thunder ports (one-time per instance, persistent across reboots):

```powershell
tnr ports forward 0 --add 8000
tnr ports forward 1 --add 8000
tnr ports list
```

Each instance now has a stable public URL: `https://<uuid>-8000.thundercompute.net`.

### Step 5 — Wire URLs and UUIDs into `scripts/redeploy.ps1`

Open [`scripts/redeploy.ps1`](scripts/redeploy.ps1) and update the `$Config` block at the top:

```powershell
$Config = @{
    HfToken    = "hf_yourtoken_here"
    Master1Url = "https://wherever-mixture-decimal-mart.trycloudflare.com"   # from Step 3 Terminal A
    Master2Url = "https://alien-rarely-could-nova.trycloudflare.com"         # from Step 3 Terminal B
    BatchSize  = 64
    ModelName  = "meta-llama/Llama-3.1-8B-Instruct"
    Instances = @(
        @{ Id = 0; Uuid = "gaxwh8fq"; WorkerPrefix = "thunder_a"; MasterTarget = "master1" },  # UUIDs from Step 4
        @{ Id = 1; Uuid = "xcidc8hb"; WorkerPrefix = "thunder_b"; MasterTarget = "master2" }
    )
}
```

You'll re-edit `Master1Url` / `Master2Url` every time you restart cloudflared (the trycloudflare URLs are random per session). UUIDs only change if you provision new instances.

### Step 6 — Deploy the Thunder workers

With the laptop stack (Step 1), test UI (Step 2), and both cloudflared tunnels (Step 3) all running:

```powershell
.\scripts\redeploy.ps1
```

The script:
1. Pings both cloudflared URLs to verify the inbound path works.
2. Pushes `thunder_worker.py` and `launch_workers.sh` to both instances in parallel via `tnr scp`.
3. Pipes the bash launcher through `tnr connect <id>` on each instance — kills any old workers, writes a per-session env script, fires uvicorn under tmux (fire-and-forget).
4. Polls `http://localhost:7001/scheduler/workers` and `:7002` every 5 s until both Thunder workers heartbeat with `slots=64`.

A full cold start (model loaded from HF cache + first heartbeat) takes ~3 minutes. Subsequent redeploys are ~30 seconds.

Add `-Force` to redeploy even when workers are already healthy (use after editing `thunder_worker.py`):

```powershell
.\scripts\redeploy.ps1 -Force
```

### Step 7 — Verify

```powershell
# Each master should show 1 CPU + 1 Thunder GPU worker, all fresh
curl.exe http://localhost:7001/scheduler/workers
curl.exe http://localhost:7002/scheduler/workers

# End-to-end through NGINX
Invoke-RestMethod -Uri "http://localhost:8008/generate" -Method Post `
  -ContentType "application/json" `
  -Body '{"prompt":"hello","max_new_tokens":32}'
```

In the test UI at [http://localhost:8050](http://localhost:8050), the **Live Workers** panel should now show all four workers:

```
master1  →  worker_1_1 (cpu)   ·  thunder_a_gpu0 (gpu, slots=64)
master2  →  worker_2_1 (cpu)   ·  thunder_b_gpu0 (gpu, slots=64)
```

With `last_seen` < 5 s on all four, and `gpu / vram` columns showing live `nvidia-smi` numbers from the Thunder workers.

---

## Smoke test

```powershell
curl http://localhost:8008/nginx/health

Invoke-RestMethod -Uri "http://localhost:8008/generate" -Method Post `
  -ContentType "application/json" `
  -Body '{"prompt":"Write a short poem about distributed systems.","max_new_tokens":128}'
```

For load testing with live throughput, p50/p95/p99 latency, per-worker breakdown, and CSV export, use the test UI at `http://localhost:8050` and click **Run Load Test**.

---

## Failover testing

**Stop a master mid-test:**
```powershell
docker compose stop master1
```
NGINX detects master1 within ~1 failed request, marks it down, and routes 100% of traffic to master2. Capacity halves; the test continues. After:
```powershell
docker compose start master1
```
traffic rebalances back to both masters.

**Kill a Thunder GPU mid-test:**
```powershell
echo "pkill -9 -f thunder_worker:app; tmux kill-session -t gpu0" | tnr connect 0
```
The master sees in-flight requests fail with `transport_RemoteProtocolError`, retries on the surviving GPU. After `STALE_AFTER_SEC` (60 s), the dead worker drops out of the registry. Bring it back with `.\scripts\redeploy.ps1 -Force`.

See [`onboard.md`](onboard.md#test-scripts) for the scripted versions of these tests.

---

## Quick reference

| Component | How to start | Where it logs | How it dies cleanly |
|---|---|---|---|
| Local stack | `docker compose up -d` | `docker compose logs <service>` | `docker compose down` |
| Test UI | `python -m client.app` | the same terminal | Ctrl-C |
| cloudflared × 2 | `cloudflared tunnel --url http://localhost:7001` (and `:7002`) | the same terminals | Ctrl-C in each terminal |
| Thunder workers | `.\scripts\redeploy.ps1` | `tmux attach -t gpu0` on the instance | `echo "tmux kill-server; pkill -9 -f thunder_worker" \| tnr connect <id>` |

| Service | Host port | Internal port |
|---|---|---|
| NGINX (public entry) | `8008` | 8008 |
| master1 | `7001` | 7000 |
| master2 | `7002` | 7000 |
| rag | — (via nginx /rag/) | 8090 |
| qdrant | `6333` | 6333 |
| cpu_worker_1_1 | `9001` | 9001 |
| cpu_worker_2_1 | `9002` | 9001 |
| client UI | `8050` | n/a (host process) |
| Thunder workers | n/a — public HTTPS via `tnr ports forward` | 8000 |

---

## Technologies Used

- **Python 3.10+** — all services and tooling
- **FastAPI + uvicorn** — masters, CPU workers, Thunder workers, RAG service, test UI
- **PyTorch + HuggingFace transformers** — LLM inference (Qwen 2.5 0.5B local, Llama 3.1 8B Instruct on Thunder)
- **httpx** — async HTTP between masters and workers
- **NGINX** — Layer-7 load balancing across master nodes (least_conn + proxy_next_upstream failover)
- **Docker Compose** — local cluster orchestration (masters, NGINX, CPU workers, RAG, Qdrant)
- **Qdrant** — vector database for RAG, with named-volume persistence
- **sentence-transformers (`all-MiniLM-L6-v2`)** — chunk + prompt embedding for RAG
- **Thunder Compute** — rented GPU instances for the GPU worker layer
- **cloudflared** — public Quick Tunnels exposing laptop masters to remote workers (heartbeats)
- **tnr CLI** — Thunder SSH wrapper with `tnr ports forward` (persistent public HTTPS to Thunder ports)
- **tmux** — keeps the Thunder worker processes alive across SSH disconnects

---

## Documentation

- [`onboard.md`](onboard.md) — full architecture, request flow, API contract, verification, ports
- [`workers/thunder/THUNDER_SETUP.md`](workers/thunder/THUNDER_SETUP.md) — comprehensive Thunder Compute walkthrough (provisioning, key permissions, port forwarding, troubleshooting)
- [`docs/master-scheduler-plan.md`](docs/master-scheduler-plan.md) — design document for the master scheduler
- [`Report.docx`](Report.docx) — full project report
