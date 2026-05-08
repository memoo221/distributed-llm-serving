# Thunder Compute GPU Workers — Full Setup

This is the comprehensive walkthrough for the Thunder GPU layer of the
distributed inference cluster. It assumes you have **2 Thunder Compute
instances, each with 2× A100 80GB GPUs**, and want to run **4 GPU workers
total** (one per physical GPU) in a symmetric layout: 2 workers heartbeat
to `master1`, 2 to `master2`.

If you have a different shape (one instance, one GPU, etc.) the same flow
applies — just run fewer workers.

---

## Architecture

```
Your laptop (NAT'd)                                  Thunder instance A (id 0, uuid q5lbf7oe)
┌──────────────────────────────────────────┐        ┌─────────────────────────────────────────┐
│ docker:                                  │        │ uvicorn :8000  thunder_worker.py        │
│   nginx :8008                            │        │   loads Llama 3.1 8B on cuda:0          │
│   master1 :7001                          │        │ uvicorn :8001  thunder_worker.py        │
│   master2 :7002                          │        │   loads Llama 3.1 8B on cuda:1          │
│   cpu_worker_1_1, cpu_worker_2_1         │        │                                         │
│                                          │        │  ports 8000/8001 exposed publicly via   │
│ host processes:                          │        │  tnr ports forward → HTTPS URLs at      │
│   client UI :8050                        │  ◄─────┤  https://q5lbf7oe-{8000,8001}.thunderc.net
│   cloudflared :7001 → trycloudflare URL  │        │                                         │
│   cloudflared :7002 → trycloudflare URL  │        │  workers heartbeat to those URLs        │
└──────────────────────────────────────────┘        └─────────────────────────────────────────┘

                                                     Thunder instance B (id 1, uuid tn9t67pp)
                                                     ┌─────────────────────────────────────────┐
                                                     │ same pattern as instance A              │
                                                     │ port 8000 → cuda:0 → master1            │
                                                     │ port 8001 → cuda:1 → master2            │
                                                     └─────────────────────────────────────────┘
```

**Why two tunnels?**
Masters live behind your laptop's NAT. Thunder workers can't reach them
directly. `cloudflared`'s anonymous Quick Tunnel exposes each master at a
public `*.trycloudflare.com` URL, which workers POST heartbeats to.

**Why `tnr ports forward` (not `tnr connect -t`)?**
Older Thunder docs used `tnr connect -t PORT` to forward Thunder ports to
your laptop's `localhost`, then masters reached workers via
`http://host.docker.internal:PORT`. The newer `tnr ports forward` exposes
Thunder ports at public HTTPS URLs (`https://<uuid>-<port>.thundercompute.net`)
that masters call directly — no laptop relay, no SSH session must stay
alive. Use this.

**Why `WORKER_DEVICE` instead of `CUDA_VISIBLE_DEVICES`?**
On Thunder's "Prototyping" mode the CUDA runtime intercepts
`CUDA_VISIBLE_DEVICES` and routes every CUDA call to the first physical
GPU. Two workers launched with `CUDA_VISIBLE_DEVICES=0` and `=1` both end
up on GPU 0. The patched `thunder_worker.py` reads a `WORKER_DEVICE` env
var (e.g. `cuda:0` / `cuda:1`) and calls `model.to(WORKER_DEVICE)`
explicitly, which PyTorch honors regardless of the masking shim.

---

## Prerequisites

On your **laptop** (Windows PowerShell):

```powershell
# 1. Thunder CLI
pip install tnr
tnr --version       # should print v2.x

# 2. Cloudflare tunnel daemon
winget install --id Cloudflare.cloudflared
cloudflared --version

# 3. Authenticate Thunder
tnr login           # opens browser
```

You also need:

- **2 running Thunder instances** with 2× A100 80GB each. Check:
  ```powershell
  tnr status
  ```
  Note the `ID` and `UUID` for each (e.g. `0 / q5lbf7oe`, `1 / tn9t67pp`)
  and confirm `Status: RUNNING`.

- **HuggingFace token** with access to the model you want to load. The
  default `meta-llama/Llama-3.1-8B-Instruct` is gated — visit
  https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and click
  "Request access" before continuing. For an open-access alternative use
  `Qwen/Qwen2.5-7B-Instruct` (set `MODEL_NAME` in the launch step).

- **Local docker stack already running** (nginx, master1, master2, the two
  CPU workers). See [`onboard.md`](../onboard.md). Verify with:
  ```powershell
  curl.exe http://localhost:7001/scheduler/workers
  curl.exe http://localhost:7002/scheduler/workers
  ```
  Each should show its CPU worker with `last_seen_sec_ago < 5`.

---

## Step 1 — Fix Windows SSH key permissions

Thunder's `tnr scp` uses OpenSSH under the hood, which refuses to use
private key files that are too permissive. On a fresh Windows install
the keys at `%USERPROFILE%\.thunder\keys\` inherit broad permissions and
will fail with `bad permissions`. Lock them down once:

```powershell
# Replace adam\adam with the output of `whoami` on your machine
$user = "$(whoami)"

foreach ($id in @("q5lbf7oe", "tn9t67pp")) {
    $key = "$env:USERPROFILE\.thunder\keys\$id"
    icacls $key /reset
    icacls $key /inheritance:r
    icacls $key /grant:r "${user}:(R)"
}
```

Verify each shows **only** your user with `(R)` and no inherited entries:

```powershell
icacls "$env:USERPROFILE\.thunder\keys\q5lbf7oe"
icacls "$env:USERPROFILE\.thunder\keys\tn9t67pp"
```

Expected:

```
C:\Users\adamt\.thunder\keys\q5lbf7oe adam\adam:(R)
Successfully processed 1 files; Failed processing 0 files
```

If `tnr scp` still complains about permissions after this, double-check
that no `(I)` (inherited) entries leaked back in.

---

## Step 2 — Push worker code to both instances

From the project root on your laptop:

```powershell
tnr scp workers/thunder_worker.py workers/thunder_requirements.txt 0:
tnr scp workers/thunder_worker.py workers/thunder_requirements.txt 1:
```

Both files land in `/home/ubuntu/` on each instance. The worker is
self-contained — it doesn't import anything from the rest of the project,
so there's no need to ship the whole repo.

> If you change `thunder_worker.py` later (e.g. to swap the model), re-run
> the same `tnr scp` commands and restart the worker processes on Thunder.

---

## Step 3 — Install Python deps and tmux on both instances

Repeat for **each instance** (`tnr connect 0` then `tnr connect 1`):

```powershell
tnr connect 0
```

Then on the Thunder shell:

```bash
# tmux is not preinstalled; apt-get update is required because the apt
# index is stale on a fresh instance
sudo apt-get update
sudo apt-get install -y tmux
tmux -V

# Sanity check: nvidia-smi should list 2x A100-SXM4-80GB
nvidia-smi --query-gpu=index,name,memory.used --format=csv

# Install Python deps for the worker
cd ~
pip install -r thunder_requirements.txt

exit
```

Total time per instance: 1–3 min depending on Thunder's network.

---

## Step 4 — Open Thunder port forwards (one-shot, persistent)

Each Thunder worker listens on a port inside the instance. To make those
ports reachable from the masters (which run in docker on your laptop),
expose them at public HTTPS URLs:

```powershell
tnr ports forward 0 --add 8000,8001
tnr ports forward 1 --add 8000,8001
tnr ports list
```

Once added these are **persistent** — they survive instance reboots and
do **not** require any open SSH session. The CLI prints URLs of the form:

```
https://q5lbf7oe-8000.thundercompute.net
https://q5lbf7oe-8001.thundercompute.net
https://tn9t67pp-8000.thundercompute.net
https://tn9t67pp-8001.thundercompute.net
```

Right now those URLs return 502 / connection-refused because no worker is
listening yet — that's fine. You'll plug them into `SELF_URL` in Step 6.

---

## Step 5 — Open cloudflared tunnels for both masters

Each master needs a public URL so Thunder workers can heartbeat home.
Use cloudflared's no-account Quick Tunnels.

**Open Terminal A (new PowerShell window). Leave it open.**

```powershell
cloudflared tunnel --url http://localhost:7001
```

After ~5 seconds it prints something like:

```
Your quick Tunnel has been created! Visit it at:
https://wherever-mixture-decimal-mart.trycloudflare.com
```

**Copy that URL — call it `MASTER1_URL`. Do NOT close this terminal.**

**Open Terminal B (another new PowerShell window). Leave it open.**

```powershell
cloudflared tunnel --url http://localhost:7002
```

Copy this URL — call it `MASTER2_URL`. Do NOT close this terminal.

Verify both tunnels reach the masters:

```powershell
curl.exe https://<MASTER1_URL>/scheduler/workers
curl.exe https://<MASTER2_URL>/scheduler/workers
```

Each should return the same JSON as `localhost:7001`/`7002`.

> The trycloudflare URLs are random and change every time you restart
> cloudflared. If you want stable URLs, sign up for a free Cloudflare
> account, claim a free domain, and use a "named tunnel" instead.

---

## Step 6 — Launch the 4 worker processes

For each instance, you'll start two tmux sessions: one running a worker
on `cuda:0` (port 8000, heartbeats to master1) and one on `cuda:1` (port
8001, heartbeats to master2).

The pattern uses `tmux send-keys` to set each env var as its own line,
which is more robust than passing a long quoted command to `tmux new`.

### 6.1 — Instance 0 (workers `thunder_a_gpu0` and `thunder_a_gpu1`)

```powershell
tnr connect 0
```

On the Thunder shell, replace `hf_yourtoken_here` with your real token
and `MASTER1_URL` / `MASTER2_URL` with the actual trycloudflare URLs:

```bash
export HF_TOKEN=hf_yourtoken_here

# Optional: confirm the token has Llama access
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  -o /dev/null -w "%{http_code}\n" \
  https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json
# 200 = good, 401 = bad token, 403 = token works but no Llama access yet

# GPU 0 → master1 → port 8000
tmux new -d -s gpu0
tmux send-keys -t gpu0 "export PYTHONUNBUFFERED=1" Enter
tmux send-keys -t gpu0 "export HF_TOKEN='$HF_TOKEN'" Enter
tmux send-keys -t gpu0 "export WORKER_DEVICE=cuda:0" Enter
tmux send-keys -t gpu0 "export WORKER_ID=thunder_a_gpu0" Enter
tmux send-keys -t gpu0 "export MASTER_URL=https://<MASTER1_URL>" Enter
tmux send-keys -t gpu0 "export SELF_URL=https://q5lbf7oe-8000.thundercompute.net" Enter
tmux send-keys -t gpu0 "uvicorn thunder_worker:app --host 0.0.0.0 --port 8000 2>&1 | tee gpu0.log" Enter

# GPU 1 → master2 → port 8001
tmux new -d -s gpu1
tmux send-keys -t gpu1 "export PYTHONUNBUFFERED=1" Enter
tmux send-keys -t gpu1 "export HF_TOKEN='$HF_TOKEN'" Enter
tmux send-keys -t gpu1 "export WORKER_DEVICE=cuda:1" Enter
tmux send-keys -t gpu1 "export WORKER_ID=thunder_a_gpu1" Enter
tmux send-keys -t gpu1 "export MASTER_URL=https://<MASTER2_URL>" Enter
tmux send-keys -t gpu1 "export SELF_URL=https://q5lbf7oe-8001.thundercompute.net" Enter
tmux send-keys -t gpu1 "uvicorn thunder_worker:app --host 0.0.0.0 --port 8001 2>&1 | tee gpu1.log" Enter

tmux ls
```

Expected `tmux ls` shows two sessions, both 1 window. The `--port` numbers
**must** match the `SELF_URL` ports (8000 with `q5lbf7oe-8000`, 8001 with
`q5lbf7oe-8001`) — getting these wrong is the most common bug.

Wait for the model to load (~2–4 minutes the first time as ~16GB downloads
from HuggingFace; subsequent worker launches share the same disk cache):

```bash
sleep 240
tail -n 5 ~/gpu0.log
tail -n 5 ~/gpu1.log
nvidia-smi --query-gpu=index,memory.used --format=csv
```

Both logs should end with:

```
[thunder_a_gpu0] loaded in NN.Ns, device=cuda:0, vram_used=~15800MB
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

`nvidia-smi` should show **both GPUs at ~16 GB** (not both on GPU 0):

```
0, 16293 MiB
1, 15871 MiB
```

If both show on GPU 0 and the other is empty, see "Both workers stuck on
GPU 0" in Troubleshooting. Once split, exit the SSH session — tmux
sessions persist server-side regardless:

```bash
exit
```

### 6.2 — Instance 1 (workers `thunder_b_gpu0` and `thunder_b_gpu1`)

Identical to Step 6.1 but with `tnr connect 1`, worker IDs `thunder_b_*`,
and `tn9t67pp` UUID URLs:

```powershell
tnr connect 1
```

```bash
sudo apt-get update          # if you didn't already
sudo apt-get install -y tmux

export HF_TOKEN=hf_yourtoken_here

tmux new -d -s gpu0
tmux send-keys -t gpu0 "export PYTHONUNBUFFERED=1" Enter
tmux send-keys -t gpu0 "export HF_TOKEN='$HF_TOKEN'" Enter
tmux send-keys -t gpu0 "export WORKER_DEVICE=cuda:0" Enter
tmux send-keys -t gpu0 "export WORKER_ID=thunder_b_gpu0" Enter
tmux send-keys -t gpu0 "export MASTER_URL=https://<MASTER1_URL>" Enter
tmux send-keys -t gpu0 "export SELF_URL=https://tn9t67pp-8000.thundercompute.net" Enter
tmux send-keys -t gpu0 "uvicorn thunder_worker:app --host 0.0.0.0 --port 8000 2>&1 | tee gpu0.log" Enter

tmux new -d -s gpu1
tmux send-keys -t gpu1 "export PYTHONUNBUFFERED=1" Enter
tmux send-keys -t gpu1 "export HF_TOKEN='$HF_TOKEN'" Enter
tmux send-keys -t gpu1 "export WORKER_DEVICE=cuda:1" Enter
tmux send-keys -t gpu1 "export WORKER_ID=thunder_b_gpu1" Enter
tmux send-keys -t gpu1 "export MASTER_URL=https://<MASTER2_URL>" Enter
tmux send-keys -t gpu1 "export SELF_URL=https://tn9t67pp-8001.thundercompute.net" Enter
tmux send-keys -t gpu1 "uvicorn thunder_worker:app --host 0.0.0.0 --port 8001 2>&1 | tee gpu1.log" Enter
```

Wait, verify with the same `tail` + `nvidia-smi`, then `exit`.

---

## Step 7 — Verify from the laptop

Each master should now show 1 CPU worker + 2 GPU workers:

```powershell
curl.exe http://localhost:7001/scheduler/workers
curl.exe http://localhost:7002/scheduler/workers
```

Master1 should list `worker_1_1` (cpu), `thunder_a_gpu0` (gpu),
`thunder_b_gpu0` (gpu). Master2 lists `worker_2_1`, `thunder_a_gpu1`,
`thunder_b_gpu1`. All `last_seen_sec_ago` should be < 10.

End-to-end through nginx:

```powershell
Invoke-RestMethod -Uri "http://localhost:8008/generate" -Method Post `
  -ContentType "application/json" `
  -Body '{"prompt":"Write a short poem about distributed systems","max_new_tokens":128}'
```

Spread test (8 requests):

```powershell
1..8 | ForEach-Object {
    try {
        $r = Invoke-RestMethod -Uri "http://localhost:8008/generate" -Method Post `
            -ContentType "application/json" `
            -Body '{"prompt":"Tell me a fun fact","max_new_tokens":256}' `
            -TimeoutSec 120
        Write-Host "[$_] $($r.master_id) -> $($r.worker_id)"
    } catch {
        Write-Host "[$_] ERROR: $($_.Exception.Message)"
    }
}
```

Expect a mix of all 4 GPU worker IDs across the 8 lines. Or use the test
UI at `http://localhost:8050` for live worker view + load testing.

---

## Configuration reference

Env vars on the Thunder worker:

| Variable | Purpose | Default | Example |
|---|---|---|---|
| `MODEL_NAME` | HF Hub id or local path | `meta-llama/Llama-3.1-8B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` |
| `WORKER_ID` | Heartbeat identity (must be unique per worker) | `thunder_worker_1` | `thunder_a_gpu0` |
| `WORKER_DEVICE` | Pin worker to a specific GPU | `cuda` (or `cpu`) | `cuda:0`, `cuda:1` |
| `SELF_URL` | URL the master uses to call back | `http://localhost:8000` | `https://q5lbf7oe-8000.thundercompute.net` |
| `MASTER_URL` | Heartbeat target (cloudflared URL) | (empty — heartbeats skipped) | `https://wherever-...trycloudflare.com` |
| `HEARTBEAT_INTERVAL` | seconds between heartbeats | `5` | |
| `HF_TOKEN` | required for gated models like Llama | n/a | `hf_xxxx` |
| `WORKER_API_KEY` | Sent as `X-API-Key` header on heartbeats | (empty) | |
| `PYTHONUNBUFFERED` | Flush stdout (recommended for live logs) | `1` recommended | |

Each worker registers with **one** master (whichever cloudflared URL you
set as `MASTER_URL`). To shift the symmetric layout, just change which
URL you point a worker at.

---

## Operational notes

### Persistent terminals required

| Terminal | What it runs | Closes if you... |
|---|---|---|
| A | `cloudflared tunnel --url http://localhost:7001` (master1) | close terminal |
| B | `cloudflared tunnel --url http://localhost:7002` (master2) | close terminal |
| (host) | `python -m client.app` (test UI — optional) | close terminal |

The Thunder side has **no** persistent local terminals. `tnr ports forward`
is server-side persistent, and tmux sessions on the Thunder instances
survive SSH disconnects (tmux runs as a daemon there).

### Recovering from disconnects

**Cloudflared dropped:** restart the tunnel and update the affected
workers' `MASTER_URL` because the trycloudflare URL changes on every
restart:

```powershell
cloudflared tunnel --url http://localhost:7001
```

Then SSH back into each worker's instance, find the affected tmux session,
and restart the worker with the new URL. Easiest is to kill and relaunch:

```bash
tmux kill-session -t gpu0
pkill -9 -f "uvicorn thunder_worker"
# then re-run the gpu0 launch block from Step 6 with the new MASTER_URL
```

**Thunder instance rebooted:** `tnr ports forward` settings persist,
but tmux sessions and the worker processes do not. SSH back in, install
tmux again if needed, and re-run the launch block.

**Local docker restarted:** masters got new container IDs and the
heartbeats keep coming, but you should restart the cloudflared tunnels so
they bind to the fresh container ports correctly.

### Stopping all workers cleanly

On each Thunder instance:

```bash
tmux kill-server
pkill -9 -f "uvicorn thunder_worker"
```

Both lines: tmux kills the wrapping session, pkill catches any process
that detached.

---

## Troubleshooting

### `bad permissions` on private key

Windows OpenSSH refuses keys readable by groups. See Step 1 and re-run
the `icacls /reset && /inheritance:r && /grant:r` block. Verify the file
ACL shows only your user, no `(I)` entries.

### `Permission denied` after fixing `icacls`

Username matched wrong. Run `whoami` and use the exact `domain\user`
form in the `/grant:r` argument (e.g. `adam\adam:(R)`).

### `tmux: command not found`

`apt-get update` is stale. Run:

```bash
sudo apt-get update && sudo apt-get install -y tmux
```

### `GatedRepoError 403` loading Llama

Your HF token is valid but your account isn't in the model's authorized
list yet. Either:

- Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct, click
  "Request access", wait for approval (usually 5–30 min).
- Or swap to an open-access model: `export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct`
  before the launch block.

### `address already in use` on uvicorn startup

Stale uvicorn process from an earlier launch survived `tmux kill-session`.
Force-kill:

```bash
pkill -9 -f "uvicorn thunder_worker"
ss -tnlp | grep -E ':(8000|8001) '   # confirm both ports free
```

Then relaunch.

### Both workers stuck on GPU 0 (one GPU shows ~31 GB, the other ~4 MiB)

Either:

1. **Patched `thunder_worker.py` not on this instance.** Check:
   ```bash
   grep WORKER_DEVICE ~/thunder_worker.py
   ```
   Should match twice (comment + `os.getenv` line). If not, re-run
   `tnr scp workers/thunder_worker.py 0:` (and `1:`) from the laptop.

2. **`WORKER_DEVICE` env var didn't reach the worker.** The `tmux send-keys`
   approach is robust, but if you get clever with one-line invocations
   make sure each `export` is its own line.

`CUDA_VISIBLE_DEVICES` will *not* fix this on Thunder — their CUDA shim
silently routes everything to the first physical GPU. `WORKER_DEVICE` +
explicit `model.to("cuda:N")` is required.

### `nvidia-smi` shows GPU memory used but "No running processes found"

Normal on Thunder. Their virtualization layer doesn't expose process IDs
to host-level `nvidia-smi`. Use `ps aux | grep [u]vicorn` to find
processes; use `nvidia-smi` only for VRAM totals.

### Master never sees the worker

- Confirm the cloudflared tunnel for that master is alive and
  `curl https://<MASTER_URL>/health` returns `nginx ok`.
- Confirm `MASTER_URL` set on the worker matches the live cloudflared
  URL (they change on every cloudflared restart).
- Look at the worker's log for `heartbeat failed` lines.

### Master sees the worker but `/generate` calls fail

Master is calling `SELF_URL`. Confirm:

- `tnr ports list` shows the port still forwarded.
- `curl https://<uuid>-8000.thundercompute.net/health` returns `{"status":"ok",...}`.
- Worker log shows it's listening on the port matching `SELF_URL`.

### First request is slow

First inference allocates CUDA kernels. Subsequent requests on the same
worker are ~3–10× faster.

### `cuda oom` on inference

Lower `max_new_tokens` in the request, or switch to a smaller model via
`MODEL_NAME`. Llama 3.1 8B in fp16 needs ~16 GB; an 80 GB A100 has
plenty of headroom for normal request sizes.

---

## Appendix — files in this layer

- `thunder_worker.py` — the FastAPI server (self-contained)
- `thunder_requirements.txt` — pip deps
- `THUNDER_SETUP.md` — this file
