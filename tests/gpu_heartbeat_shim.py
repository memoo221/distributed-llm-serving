"""
Heartbeat shim: register a remote Kaggle GPU worker with a local master.

Why this exists: the GPU worker runs on Kaggle and can't reach the local master
to send its own heartbeats. This shim runs on the host, polls the worker's
public /health endpoint, and re-posts the payload to the master so the master's
WorkerRegistry treats the GPU as live and routable.

Run from project root:

    python tests/gpu_heartbeat_shim.py \
        --worker-url https://unified-flirt-switch.ngrok-free.dev/w1 \
        --worker-id worker_gpu_1 \
        --master-url http://localhost:7001 \
        --interval 5
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

import httpx


def beat_once(client: httpx.Client, worker_url: str, worker_id: str, master_url: str) -> str:
    health = client.get(f"{worker_url}/health", timeout=5.0)
    health.raise_for_status()
    h = health.json()

    payload = {
        "worker_id": worker_id,
        "url": worker_url,
        "device_type": "gpu",
        "active_requests": int(h.get("active_requests", 0)),
        "queue_depth": int(h.get("queue_depth", 0)),
        "gpu_util_pct": h.get("gpu_util_pct"),
        "vram_used_mb": h.get("vram_used_mb"),
        "vram_total_mb": h.get("vram_total_mb"),
        "total_requests": h.get("total_requests"),
        "total_errors": h.get("total_errors"),
        "batching": h.get("batching"),
        "batch_size": h.get("batch_size"),
        "timestamp": time.time(),
    }

    r = client.post(f"{master_url.rstrip('/')}/heartbeat", json=payload, timeout=5.0)
    r.raise_for_status()
    return (
        f"active={payload['active_requests']} util={payload['gpu_util_pct']}% "
        f"total={payload['total_requests']}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--worker-url", required=True)
    p.add_argument("--worker-id", default="worker_gpu_1")
    p.add_argument("--master-url", default="http://localhost:7001")
    p.add_argument("--interval", type=float, default=5.0)
    args = p.parse_args()

    print(f"[shim] {args.worker_id} :: {args.worker_url} -> {args.master_url} every {args.interval}s")

    stop = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(v=True))
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, lambda *_: stop.update(v=True))

    with httpx.Client() as client:
        while not stop["v"]:
            t = time.strftime("%H:%M:%S")
            try:
                summary = beat_once(client, args.worker_url, args.worker_id, args.master_url)
                print(f"[shim {t}] OK  {summary}", flush=True)
            except Exception as e:
                print(f"[shim {t}] ERR {type(e).__name__}: {e}", flush=True)
            for _ in range(int(args.interval * 10)):
                if stop["v"]:
                    break
                time.sleep(0.1)

    print("[shim] stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
