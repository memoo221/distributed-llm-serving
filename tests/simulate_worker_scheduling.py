"""
Simulate concurrent /generate traffic against a single master and tally which
workers handled each request. Useful for sanity-checking the scheduler's
threshold routing (small → CPU, large → GPU) and CPU-saturation fallback.

Run from project root:

    # 20 requests, 8 in flight, 30% large (GPU-bound)
    python tests/simulate_worker_scheduling.py --base-url http://localhost:7001

    # All small — exercises CPU pool + saturation fallback to GPU
    python tests/simulate_worker_scheduling.py --large-ratio 0

    # All large — should pin to groq workers
    python tests/simulate_worker_scheduling.py --large-ratio 1

    # Paced load (1 req/s) to stay within Groq free-tier RPM ceiling
    python tests/simulate_worker_scheduling.py --requests 20 --concurrency 1 --delay 1.0

    # Save results to CSV
    python tests/simulate_worker_scheduling.py --requests 100 --csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import time
from statistics import mean, median
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


SMALL_TOKENS = 32   # score ~= 35 (3 prompt tokens + 32), below GPU_ROUTE_THRESHOLD=256 → CPU first
LARGE_TOKENS = 300  # score ~= 303, >= 256 → GPU only


def _stringify_error(error: Any) -> str:
    if error is None:
        return ""
    if isinstance(error, str):
        return error
    return str(error)


def _percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _is_gpu_worker(worker_id: str | None) -> bool:
    return bool(worker_id and "groq" in worker_id)


async def fire_one(
    client: httpx.AsyncClient,
    base_url: str,
    index: int,
    is_large: bool,
    delay_before: float,
) -> dict[str, Any]:
    if delay_before > 0:
        await asyncio.sleep(delay_before)

    max_new_tokens = LARGE_TOKENS if is_large else SMALL_TOKENS
    payload = {"prompt": f"req#{index}: say hi", "max_new_tokens": max_new_tokens}
    t0 = time.monotonic()
    try:
        resp = await client.post(f"{base_url}/generate", json=payload)
        elapsed = time.monotonic() - t0
        body = resp.json() if "application/json" in resp.headers.get("content-type", "") else {}
        return {
            "index": index,
            "kind": "large" if is_large else "small",
            "status": resp.status_code,
            "worker_id": body.get("worker_id"),
            "elapsed_s": round(elapsed, 2),
            "error": None if resp.status_code == 200 else (body or resp.text),
        }
    except httpx.TimeoutException as exc:
        elapsed = time.monotonic() - t0
        return {
            "index": index,
            "kind": "large" if is_large else "small",
            "status": None,
            "worker_id": None,
            "elapsed_s": round(elapsed, 2),
            "error": f"timeout after {elapsed:.1f}s: {type(exc).__name__}",
        }
    except httpx.ConnectError as exc:
        elapsed = time.monotonic() - t0
        return {
            "index": index,
            "kind": "large" if is_large else "small",
            "status": None,
            "worker_id": None,
            "elapsed_s": round(elapsed, 2),
            "error": f"connect error: {exc}",
        }
    except httpx.HTTPError as exc:
        elapsed = time.monotonic() - t0
        return {
            "index": index,
            "kind": "large" if is_large else "small",
            "status": None,
            "worker_id": None,
            "elapsed_s": round(elapsed, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }


async def run(
    base_url: str,
    total: int,
    concurrency: int,
    large_ratio: float,
    seed: int,
    delay: float,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    kinds = [rng.random() < large_ratio for _ in range(total)]

    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(timeout=240.0, limits=limits) as client:
        async def guarded(i: int) -> dict[str, Any]:
            async with sem:
                # Stagger requests within each concurrency slot to spread load
                slot_delay = delay * (i % concurrency) / concurrency if delay > 0 else 0
                return await fire_one(client, base_url, i, kinds[i - 1], slot_delay)

        return await asyncio.gather(*[guarded(i) for i in range(1, total + 1)])


def save_stats_to_csv(
    results: list[dict[str, Any]],
    base_url: str,
    total: int,
    concurrency: int,
    large_ratio: float,
    run_duration_s: float,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = Path("logs")
    csv_dir.mkdir(exist_ok=True)

    requests_csv = csv_dir / f"requests_{timestamp}.csv"
    with open(requests_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "kind", "status", "worker_id", "elapsed_s", "error"])
        writer.writeheader()
        writer.writerows(results)

    by_worker = Counter(r["worker_id"] for r in results if r["status"] == 200)
    by_kind_worker = Counter((r["kind"], r["worker_id"]) for r in results if r["status"] == 200)
    failures = [r for r in results if r["status"] != 200]
    successful = len(results) - len(failures)
    latencies = [float(r["elapsed_s"]) for r in results if r["status"] == 200 and r.get("elapsed_s") is not None]

    small_to_gpu = sum(c for (k, w), c in by_kind_worker.items() if k == "small" and w and "groq" in w)
    large_to_cpu = sum(c for (k, w), c in by_kind_worker.items() if k == "large" and w and "groq" not in w)
    cpu_workers = len({r["worker_id"] for r in results if r["status"] == 200 and not _is_gpu_worker(r["worker_id"])})
    gpu_workers = len({r["worker_id"] for r in results if r["status"] == 200 and _is_gpu_worker(r["worker_id"])})
    cpu_busy_s = sum(float(r["elapsed_s"]) for r in results if r["status"] == 200 and not _is_gpu_worker(r["worker_id"]))
    gpu_busy_s = sum(float(r["elapsed_s"]) for r in results if r["status"] == 200 and _is_gpu_worker(r["worker_id"]))
    wall_time = max(run_duration_s, 1e-9)
    cpu_utilization = min(100.0, (cpu_busy_s / (cpu_workers * wall_time) * 100.0)) if cpu_workers else 0.0
    gpu_utilization = min(100.0, (gpu_busy_s / (gpu_workers * wall_time) * 100.0)) if gpu_workers else 0.0
    avg_latency = mean(latencies) if latencies else 0.0
    median_latency = median(latencies) if latencies else 0.0
    p95_latency = _percentile(latencies, 0.95)
    throughput = successful / wall_time

    timeout_errors = sum(1 for r in failures if "timeout" in _stringify_error(r["error"]).lower())
    server_503 = sum(1 for r in failures if r["status"] == 503)

    stats_csv = csv_dir / f"stats_{timestamp}.csv"
    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Base URL", base_url])
        writer.writerow(["Total Requests", total])
        writer.writerow(["Concurrency", concurrency])
        writer.writerow(["Large Ratio", large_ratio])
        writer.writerow(["Run Duration (s)", round(run_duration_s, 2)])
        writer.writerow(["Throughput (successful req/s)", round(throughput, 2)])
        writer.writerow(["Average Latency (s)", round(avg_latency, 2)])
        writer.writerow(["Median Latency (s)", round(median_latency, 2)])
        writer.writerow(["P95 Latency (s)", round(p95_latency, 2)])
        writer.writerow(["Observed CPU Utilization (%)", round(cpu_utilization, 1)])
        writer.writerow(["Observed GPU Utilization (%)", round(gpu_utilization, 1)])
        writer.writerow(["Successful Requests", successful])
        writer.writerow(["Failed Requests", len(failures)])
        writer.writerow(["  → Client timeouts", timeout_errors])
        writer.writerow(["  → Server 503s", server_503])
        writer.writerow(["Small→GPU (Saturation)", small_to_gpu])
        writer.writerow(["Large→CPU (Should be 0)", large_to_cpu])
        writer.writerow([""])
        writer.writerow(["Worker Distribution", "Count"])
        for w, c in sorted(by_worker.items()):
            writer.writerow([w, c])
        writer.writerow([""])
        writer.writerow(["Kind→Worker Distribution", "Count"])
        for (kind, w), c in sorted(by_kind_worker.items()):
            writer.writerow([f"{kind}→{w}", c])

    return str(stats_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test a master's worker scheduling.")
    parser.add_argument("--base-url", default="http://localhost:7001", help="Master base URL (default: master1)")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--large-ratio", type=float, default=0.3,
                        help="Fraction of requests that should force GPU routing (max_new_tokens=300)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to stagger between each concurrency slot (e.g. 1.0 for ~1 req/s per slot)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducible large/small mix (default: 0)")
    parser.add_argument("--csv", action="store_true", help="Save per-request and aggregate results to logs/")
    args = parser.parse_args()

    started_at = time.monotonic()

    results = asyncio.run(run(
        base_url=args.base_url.rstrip("/"),
        total=args.requests,
        concurrency=args.concurrency,
        large_ratio=args.large_ratio,
        seed=args.seed,
        delay=args.delay,
    ))

    print("Per-request results:")
    for r in results:
        if r["status"] == 200:
            print(f"  req={r['index']:03d} kind={r['kind']:5s} status=200 worker={r['worker_id']} ({r['elapsed_s']}s)")
        else:
            print(f"  req={r['index']:03d} kind={r['kind']:5s} status={r['status']} error={r['error']}")

    by_worker = Counter(r["worker_id"] for r in results if r["status"] == 200)
    by_kind_worker = Counter((r["kind"], r["worker_id"]) for r in results if r["status"] == 200)
    failures = [r for r in results if r["status"] != 200]
    successful = len(results) - len(failures)

    print("\nDistribution by worker:")
    for w, c in sorted(by_worker.items()):
        print(f"  {w}: {c}")

    print("\nDistribution by (kind, worker):")
    for (kind, w), c in sorted(by_kind_worker.items()):
        print(f"  {kind:5s} → {w}: {c}")

    print("\nSanity checks:")
    small_to_gpu = sum(c for (k, w), c in by_kind_worker.items() if k == "small" and w and "groq" in w)
    large_to_cpu = sum(c for (k, w), c in by_kind_worker.items() if k == "large" and w and "groq" not in w)
    print(f"  small requests routed to GPU (saturation fallback): {small_to_gpu}")
    print(f"  large requests routed to CPU (should be 0):          {large_to_cpu}")

    timeout_errors = sum(1 for r in failures if "timeout" in _stringify_error(r["error"]).lower())
    server_503 = sum(1 for r in failures if r["status"] == 503)
    latencies = [float(r["elapsed_s"]) for r in results if r["status"] == 200 and r.get("elapsed_s") is not None]
    avg_latency = mean(latencies) if latencies else 0.0
    median_latency = median(latencies) if latencies else 0.0
    p95_latency = _percentile(latencies, 0.95)
    duration_s = max(time.monotonic() - started_at, 1e-9)
    throughput = successful / duration_s
    cpu_workers = len({r["worker_id"] for r in results if r["status"] == 200 and not _is_gpu_worker(r["worker_id"])})
    gpu_workers = len({r["worker_id"] for r in results if r["status"] == 200 and _is_gpu_worker(r["worker_id"])})
    cpu_busy_s = sum(float(r["elapsed_s"]) for r in results if r["status"] == 200 and not _is_gpu_worker(r["worker_id"]))
    gpu_busy_s = sum(float(r["elapsed_s"]) for r in results if r["status"] == 200 and _is_gpu_worker(r["worker_id"]))
    cpu_utilization = min(100.0, (cpu_busy_s / (cpu_workers * duration_s) * 100.0)) if cpu_workers else 0.0
    gpu_utilization = min(100.0, (gpu_busy_s / (gpu_workers * duration_s) * 100.0)) if gpu_workers else 0.0
    print(f"\nResults: {successful}/{len(results)} succeeded ({100*successful//len(results)}%)")
    print(f"  throughput: {throughput:.2f} req/s")
    print(f"  latency: avg={avg_latency:.2f}s median={median_latency:.2f}s p95={p95_latency:.2f}s")
    print(f"  cpu utilization: {cpu_utilization:.1f}%")
    print(f"  gpu utilization: {gpu_utilization:.1f}%")
    if failures:
        print(f"Failures: {len(failures)} total")
        if timeout_errors:
            print(f"  client timeouts:  {timeout_errors}  ← concurrency too high or Groq RPM exhausted")
        if server_503:
            print(f"  server 503s:      {server_503}  ← master queue wait timed out (all workers busy/cooling)")
        other = len(failures) - timeout_errors - server_503
        if other:
            print(f"  other errors:     {other}")

    if args.csv:
        csv_file = save_stats_to_csv(
            results,
            args.base_url,
            args.requests,
            args.concurrency,
            args.large_ratio,
            duration_s,
        )
        print(f"\n✓ Stats saved to: {csv_file}")


if __name__ == "__main__":
    main()