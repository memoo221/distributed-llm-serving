"""
Smoke test for WorkerNode heartbeats.

Spins up a CPU-mode WorkerNode (no model load — uses a stubbed remote so we
don't pay tokenizer/model load just to test heartbeats), points its heartbeat
loop at master1's exposed port (default 7001), waits long enough to capture a
few sends, then verifies the master saw them via GET /heartbeat/workers.

Prerequisites:
    docker compose up --build -d        # rebuild after master mock_api changes
    # master1 must publish 7001:7000 in docker-compose.yml

Run from project root:

    python tests/smoke_heartbeat.py
    python tests/smoke_heartbeat.py --master-url http://localhost:7001 --interval 2 --duration 7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workers.worker_service import WorkerNode  # noqa: E402


def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _result(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for WorkerNode heartbeats")
    parser.add_argument(
        "--master-url",
        default="http://localhost:7001",
        help="Master to send heartbeats to (default master1 on 7001)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Heartbeat interval in seconds (default 2)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=7.0,
        help="How long to keep the heartbeat running before stopping (default 7s)",
    )
    args = parser.parse_args()

    overall_ok = True

    # 1) Pre-flight: master reachable?
    _section("Pre-flight")
    try:
        r = httpx.get(f"{args.master_url}/health", timeout=3.0)
        r.raise_for_status()
        _result("master reachable", True, f"{args.master_url} → {r.json().get('master_id')}")
    except Exception as e:
        _result(
            "master reachable",
            False,
            f"{args.master_url} not reachable: {type(e).__name__}: {e}",
        )
        print("  Hint: run `docker compose up --build -d` and confirm master1 publishes 7001:7000")
        return 1

    # 2) Create a WorkerNode in cuda mode (stub remote — we won't call /generate;
    #    we only care about heartbeats). cuda mode skips local model loading,
    #    keeping this test fast.
    _section("Start heartbeat")
    node = WorkerNode(
        model_path="remote-only",
        device="cuda:0",
        remote_endpoint="http://stub-not-used",
        worker_id="hb-test-1",
    )
    node.start_heartbeat(args.master_url, interval_sec=args.interval)

    # 3) Wait long enough to capture multiple sends.
    expected_min_sends = max(1, int(args.duration // args.interval))
    print(f"  Sleeping {args.duration}s, expecting at least {expected_min_sends} heartbeat(s)...")
    time.sleep(args.duration)

    # 4) Stop heartbeat and verify master saw the worker.
    _section("Verify on master")
    node.stop_heartbeat()

    try:
        r = httpx.get(f"{args.master_url}/heartbeat/workers", timeout=3.0)
        r.raise_for_status()
        data = r.json()
        workers = data.get("workers", {})
        seen = workers.get(node.worker_id)
        ok = seen is not None
        _result(
            f"worker_id={node.worker_id} seen by master",
            ok,
            f"device_type={seen.get('device_type') if seen else 'n/a'}, "
            f"last_total_requests={seen.get('total_requests') if seen else 'n/a'}",
        )
        overall_ok &= ok
        print(f"  Master ack: master_id={data.get('master_id')}, total_workers_seen={data.get('count')}")
    except Exception as e:
        _result("GET /heartbeat/workers", False, f"{type(e).__name__}: {e}")
        overall_ok = False

    _section("SUMMARY")
    print(f"  Overall: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
