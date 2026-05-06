# Thunder Compute GPU Worker

A FastAPI server that runs on a Thunder Compute instance, loads a model on
the GPU, and registers itself with a master via heartbeats.

## Files

- `thunder_worker.py` — the server (self-contained, no project imports)
- `thunder_requirements.txt` — Python deps

## Setup on the Thunder instance

```bash
# (after `tnr scp` finishes)
tnr connect 0     # SSH into instance 0

cd ~/distributed-llm-serving/workers
pip install -r thunder_requirements.txt
```

First model load downloads from HF Hub (~1GB for Qwen 2.5 0.5B).

## Run

The worker needs three env vars: which master to heartbeat, what URL the
master should call back on, and (optionally) which model to load.

```bash
export MASTER_URL=https://<your-tunnel>.trycloudflare.com
export SELF_URL=http://<thunder-public-ip>:8000
export WORKER_ID=thunder_worker_1
# optional — defaults to Qwen/Qwen2.5-0.5B-Instruct
# export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct

uvicorn thunder_worker:app --host 0.0.0.0 --port 8000
```

The first log line should be `loading <model> on cuda`. Once you see
`vram_used=XXXMB` and the heartbeat starts firing, the master should pick it
up within `WORKER_STALE_SEC` (default 15s).

## Verifying

From your laptop:

```bash
# Health on the worker directly
curl http://<thunder-public-ip>:8000/health

# What the master sees
curl https://<your-tunnel>.trycloudflare.com/scheduler/workers
```

You should see `thunder_worker_1` listed with `device_type: "gpu"`.

## Background it on the instance

`uvicorn` in the foreground dies when SSH disconnects. To keep it running:

```bash
# nohup is the simplest
nohup uvicorn thunder_worker:app --host 0.0.0.0 --port 8000 > worker.log 2>&1 &

# Or use tmux / screen if you want to reattach later
tmux new -s worker
# inside tmux: run uvicorn, then Ctrl+B, D to detach
```

## Configuration

| env var               | default                       | notes                                  |
|-----------------------|-------------------------------|----------------------------------------|
| `MODEL_NAME`          | `Qwen/Qwen2.5-0.5B-Instruct`  | HF Hub id or local path                |
| `WORKER_ID`           | `thunder_worker_1`            | unique per worker; master keys on it   |
| `SELF_URL`            | `http://localhost:8000`       | URL the master uses to call back       |
| `MASTER_URL`          | (empty — heartbeats skipped)  | full URL incl. https://                |
| `HEARTBEAT_INTERVAL`  | `5`                           | seconds                                |
| `WORKER_API_KEY`      | (empty)                       | sent as `X-API-Key` if set             |

## Troubleshooting

- **`CUDA out of memory` on load**: model too big for the rented GPU. Use a
  smaller `MODEL_NAME` (e.g. `Qwen/Qwen2.5-0.5B-Instruct`) or rent a bigger
  GPU. Llama-3.1-8B in fp16 needs ~16GB; in 4-bit ~6GB.
- **Master never sees the worker**: check that `MASTER_URL` is reachable
  *from the Thunder instance* (`curl $MASTER_URL/health`). If you're using a
  trycloudflare tunnel, confirm the cloudflared process is still up on your
  laptop.
- **Master sees worker but `/generate` calls fail**: master is calling
  `SELF_URL` — must be reachable *from the master*, i.e. the Thunder
  instance's public IP + the port uvicorn is listening on. Check Thunder's
  firewall settings.
- **First request is slow**: first inference allocates CUDA kernels. Subsequent
  requests should be much faster.
