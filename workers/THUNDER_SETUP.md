# Thunder Compute GPU Workers — Full Setup

This is the comprehensive walkthrough for the Thunder GPU layer of the
distributed inference cluster. It assumes you have **2 Thunder Compute
instances, each with 2× A100 80GB GPUs**, and want to run **2 GPU workers
total** (one per instance, on `cuda:0`), with each worker registering to a
different master so the cluster stays symmetric.

> **Why one worker per instance, not two?** We tried running a second worker
> per instance pinned to `cuda:1`. Thunder's "Prototyping" mode CUDA
> virtualization consistently stalled the second process — health checks
> succeeded but `model.generate()` never returned. With only `cuda:0`
> exercised per instance, all workers serve traffic correctly. The second
> A100 per instance is paid-for-but-idle until Thunder fixes the
> virtualization (or you switch to "Production" mode / different provider).

If you have a different shape (one instance only, etc.), `scripts/redeploy.ps1`
adapts — just edit the `$Config.Instances` list.

---

## Architecture

```
Your laptop (NAT'd)                                  Thunder instance A (id 0, uuid q5lbf7oe)
┌──────────────────────────────────────────┐        ┌─────────────────────────────────────────┐
│ docker:                                  │        │ uvicorn :8000  thunder_worker.py        │
│   nginx :8008                            │        │   loads Llama 3.1 8B on cuda:0          │
│   master1 :7001                          │        │   batches concurrent /generate calls    │
│   master2 :7002                          │        │   (BATCH_SIZE=64 by default)            │
│   cpu_worker_1_1, cpu_worker_2_1         │        │                                         │
│                                          │        │   port 8000 exposed via                 │
│ host processes:                          │        │   tnr ports forward → HTTPS at          │
│   client UI :8050                        │  ◄─────┤   https://q5lbf7oe-8000.thundercompute.net
│   cloudflared :7001 → trycloudflare URL  │        │                                         │
│   cloudflared :7002 → trycloudflare URL  │        │   heartbeats to MASTER1_URL             │
└──────────────────────────────────────────┘        └─────────────────────────────────────────┘

                                                     Thunder instance B (id 1, uuid tn9t67pp)
                                                     ┌─────────────────────────────────────────┐
                                                     │ same pattern as instance A              │
                                                     │ port 8000 → cuda:0                      │
                                                     │ heartbeats to MASTER2_URL               │
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

**Continuous batching.**
Each worker runs a single `_batcher_loop` task that pulls concurrent
`/generate` requests off an internal `asyncio.Queue` and coalesces up to
`BATCH_SIZE` of them (default 64) into one `model.generate()` call. The
batcher waits up to `BATCH_WAIT_MS` (default 30 ms) after the first
request arrives to gather more before flushing. This lets multiple
prompts share each forward pass on the GPU, multiplying effective
throughput.

The worker advertises `BATCH_SIZE` as its `slots` value in the heartbeat,
so the master knows it can dispatch that many concurrent requests per
worker. The master's `WorkerState.slots` honors this per-worker override
(falling back to global `GPU_SLOTS` for workers that don't advertise).
Empirically, `BATCH_SIZE=64` on Llama 3.1 8B fp16 / A100 80GB delivers
~6 req/s per worker at p50 ~30s under saturation; raising to 128 returns
diminishing throughput at the cost of higher tail latency.

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

## Step 6 — Launch the 2 worker processes

The fastest path is the [`scripts/redeploy.ps1`](../scripts/redeploy.ps1)
driver — it pushes code, kills any old workers, launches new ones via
`bash launch_workers.sh` on each instance (in parallel), and polls master
registries until all workers heartbeat.

### 6.1 — Edit the redeploy config (one-time)

Open `scripts/redeploy.ps1` and update the `$Config` block:

```powershell
$Config = @{
    HfToken    = "hf_yourtoken_here"
    Master1Url = "https://wherever-mixture-decimal-mart.trycloudflare.com"
    Master2Url = "https://alien-rarely-could-nova.trycloudflare.com"
    BatchSize  = 64
    ModelName  = "meta-llama/Llama-3.1-8B-Instruct"
    Instances = @(
        @{ Id = 0; Uuid = "q5lbf7oe"; WorkerPrefix = "thunder_a"; MasterTarget = "master1" },
        @{ Id = 1; Uuid = "tn9t67pp"; WorkerPrefix = "thunder_b"; MasterTarget = "master2" }
    )
}
```

You'll re-edit `Master1Url` / `Master2Url` whenever the cloudflared tunnels
restart (the trycloudflare URLs are random).

### 6.2 — Run the redeploy

```powershell
.\scripts\redeploy.ps1
```

Output:

```
==> Verifying cloudflared tunnels and masters
    OK: master1 reachable via https://...
    OK: master2 reachable via https://...

==> Checking if workers are already healthy
    OK: both thunder workers live with slots=64 - skipping redeploy
```

(Idempotent fast-path skips the redeploy entirely if everything's already
running with the right config — typical run-time: ~3 sec.)

If something needs to change, force the redeploy:

```powershell
.\scripts\redeploy.ps1 -Force
```

That will:

1. **Push files** — `tnr scp workers/thunder_worker.py` and
   `workers/launch_workers.sh` to each instance, **in parallel** via
   PowerShell `Start-Job`.
2. **Run the bash launcher** — pipes the kill-old-then-launch-new script
   through `tnr connect <id>` (`stdin` works for piping commands; tnr's
   own SSH wrapper is what authenticates).
3. **Wait** — polls `http://localhost:7001/scheduler/workers` and
   `:7002` every 5 sec, prints status until each worker shows up with
   `slots: BATCH_SIZE` and `last_seen_sec_ago < 10` (or fails after 6
   min).

A full cold start (model to cuda:0 from HF cache + heartbeat) takes
~3 minutes wall-clock for both instances combined.

### 6.3 — What `launch_workers.sh` actually does

The bash script runs on each Thunder instance and:

1. Bootstraps `tmux` if missing (`apt-get install -y tmux`).
2. `tmux kill-server` + `pkill -9 -f "uvicorn thunder_worker"` to clear
   any previous workers.
3. Removes stale `~/gpu0.log` so subsequent log polling doesn't match
   old "Application startup complete" lines.
4. Writes `/tmp/gpu0_launch.sh` — a self-contained bash script with all
   the env vars (`WORKER_DEVICE=cuda:0`, `MASTER_URL=...` matching
   `MASTER_TARGET`, `BATCH_SIZE`, etc.) and the `uvicorn` command.
5. `tmux new -d -s gpu0 'bash /tmp/gpu0_launch.sh'` — fire-and-forget.
   The tmux session persists server-side regardless of SSH state.

Variables are *baked into a script file* rather than sent via
`tmux send-keys` because keystroke-based env injection had a real race
where individual `export` lines could be dropped under load — see
"Variables not landing in tmux" in Troubleshooting.

### 6.4 — Manual launch (if you want to skip the script)

If you'd rather do it by hand on one instance:

```powershell
tnr connect 0
```

```bash
sudo apt-get update && sudo apt-get install -y tmux
export HF_TOKEN=hf_yourtoken_here

tmux kill-server 2>/dev/null
pkill -9 -f "uvicorn thunder_worker" 2>/dev/null
rm -f ~/gpu0.log
sleep 3

cat > /tmp/gpu0_launch.sh <<EOF
#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
export HF_TOKEN='$HF_TOKEN'
export WORKER_DEVICE='cuda:0'
export BATCH_SIZE='64'
export WORKER_ID='thunder_a_gpu0'
export MASTER_URL='https://<MASTER1_URL>'
export SELF_URL='https://q5lbf7oe-8000.thundercompute.net'
export MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
cd /home/ubuntu
exec uvicorn thunder_worker:app --host 0.0.0.0 --port 8000 2>&1 | tee /home/ubuntu/gpu0.log
EOF
chmod +x /tmp/gpu0_launch.sh
tmux new -d -s gpu0 'bash /tmp/gpu0_launch.sh'

# Watch model load (~2-3 min on Thunder for fp16 .to(cuda:0))
tail -f ~/gpu0.log    # Ctrl+C when "Application startup complete" appears
```

For instance 1, swap `WORKER_ID=thunder_b_gpu0`, `MASTER_URL=<MASTER2_URL>`,
`SELF_URL=https://tn9t67pp-8000.thundercompute.net`.

---

## Step 7 — Verify from the laptop

Each master should now show 1 CPU worker + 1 GPU worker:

```powershell
curl.exe http://localhost:7001/scheduler/workers
curl.exe http://localhost:7002/scheduler/workers
```

Master1 should list `worker_1_1` (cpu) and `thunder_a_gpu0` (gpu).
Master2 should list `worker_2_1` and `thunder_b_gpu0`. All
`last_seen_sec_ago` should be < 10. Stale entries from previous
configurations may also appear; restart the masters
(`docker compose restart master1 master2`) to clean the registry.

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
| `TEMPERATURE` | Sampling temperature for generation | `0.7` | `0.9` |
| `TOP_P` | Nucleus sampling cutoff | `0.9` | `0.95` |
| `BATCH_SIZE` | Max requests coalesced into one `model.generate()` call. Also advertised to the master as the worker's `slots`, so the master sends up to this many concurrent requests per worker | `64` | `128` |
| `BATCH_WAIT_MS` | After receiving the first request, how long the batcher waits for more requests to arrive before flushing | `30` | `50` |
| `MASTER_TARGET` | Used by `launch_workers.sh` to pick which `MASTER{1,2}_URL` the worker registers to | `master1` | `master1` or `master2` |

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

### Worker on `cuda:1` hangs / `model.generate` never returns

Thunder's "Prototyping" mode CUDA virtualization stalls a second worker
process on the same instance, even when pinned to a different GPU via
`WORKER_DEVICE=cuda:1`. Symptoms: `/health` returns OK with
`active_requests` climbing, but direct curl to `/generate` times out.

**This is why the default config runs only ONE worker per instance** (on
`cuda:0`). The second A100 is left idle. We tried various workarounds —
sequential vs. parallel launch, `BATCH_SIZE` lowering, `pkill -9` resets
— none cleared the stall reliably. The only fix was rebooting the
instance, but `sudo reboot` doesn't work on Thunder's container shell
and `tnr modify` is currently disabled.

If you need the second GPU on each instance, options:
- Switch to Thunder "Production" mode (might allow real concurrent CUDA)
- Use tensor parallelism via vLLM so one process spans both GPUs
- Move to a different cloud GPU provider

### Variables not landing in tmux (`MODEL_NAME=''` / partial env)

Older versions of `launch_workers.sh` used `tmux send-keys` to type each
`export` line into a session. Under load, individual keystrokes could be
dropped — we observed `MODEL_NAME` getting eaten on the second worker
launched on the same instance, leading to `OSError: Repo id ... cannot
be ''`.

The current `launch_workers.sh` writes `/tmp/gpu0_launch.sh` containing
all env vars, then `tmux new -d -s gpu0 'bash /tmp/gpu0_launch.sh'` —
single atomic script execution, no keystroke race. If you ever revert to
`send-keys`, expect intermittent variable loss.

### `Patched thunder_worker.py not on this instance`

Quick verification:

```bash
grep -E "WORKER_DEVICE|BATCH_SIZE" ~/thunder_worker.py
```

Should print at least 4 matches (env var declarations + comments). If
not, the file is stale — `scripts/redeploy.ps1` re-uploads on every run,
or use `tnr scp workers/thunder_worker.py 0:` manually.

`CUDA_VISIBLE_DEVICES` will *not* work on Thunder — their CUDA shim
silently routes everything to the first physical GPU. `WORKER_DEVICE` +
explicit `model.to("cuda:N")` is required for any GPU pinning.

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

- `thunder_worker.py` — the FastAPI server (self-contained, no project imports)
- `thunder_requirements.txt` — pip deps for the Thunder side
- `launch_workers.sh` — bash launcher uploaded to each Thunder instance: kills old workers, writes a per-session env script to `/tmp`, kicks off uvicorn under `tmux` (fire-and-forget)
- `THUNDER_SETUP.md` — this file
- `../scripts/redeploy.ps1` — laptop-side driver that scps both files to both instances in parallel, runs `launch_workers.sh`, and polls master registries until workers heartbeat
