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
"""

from __future__ import annotations

import argparse
import asyncio
import random
from collections import Counter
from typing import Any

import httpx


SMALL_TOKENS = 32   # score ~= 32, below GPU_ROUTE_THRESHOLD=256 → CPU first
LARGE_TOKENS = 300  # score >= 256 → GPU only


async def fire_one(
    client: httpx.AsyncClient,
    base_url: str,
    index: int,
    is_large: bool,
) -> dict[str, Any]:
    max_new_tokens = LARGE_TOKENS if is_large else SMALL_TOKENS
    payload = {"prompt": f"req#{index}: say hi", "max_new_tokens": max_new_tokens}
    try:
        resp = await client.post(f"{base_url}/generate", json=payload)
        body = resp.json() if "application/json" in resp.headers.get("content-type", "") else {}
        return {
            "index": index,
            "kind": "large" if is_large else "small",
            "status": resp.status_code,
            "worker_id": body.get("worker_id"),
            "error": None if resp.status_code == 200 else (body or resp.text),
        }
    except httpx.HTTPError as exc:
        return {
            "index": index,
            "kind": "large" if is_large else "small",
            "status": None,
            "worker_id": None,
            "error": str(exc),
        }


async def run(
    base_url: str,
    total: int,
    concurrency: int,
    large_ratio: float,
) -> list[dict[str, Any]]:
    rng = random.Random(0)
    kinds = [rng.random() < large_ratio for _ in range(total)]

    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(timeout=60.0, limits=limits) as client:
        async def guarded(i: int) -> dict[str, Any]:
            async with sem:
                return await fire_one(client, base_url, i, kinds[i - 1])

        return await asyncio.gather(*[guarded(i) for i in range(1, total + 1)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test a master's worker scheduling.")
    parser.add_argument("--base-url", default="http://localhost:7001", help="Master base URL (default: master1)")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--large-ratio", type=float, default=0.3,
                        help="Fraction of requests that should force GPU routing (max_new_tokens=300)")
    args = parser.parse_args()

    results = asyncio.run(run(
        base_url=args.base_url.rstrip("/"),
        total=args.requests,
        concurrency=args.concurrency,
        large_ratio=args.large_ratio,
    ))

    print("Per-request results:")
    for r in results:
        if r["status"] == 200:
            print(f"  req={r['index']:02d} kind={r['kind']:5s} status=200 worker={r['worker_id']}")
        else:
            print(f"  req={r['index']:02d} kind={r['kind']:5s} status={r['status']} error={r['error']}")

    by_worker = Counter(r["worker_id"] for r in results if r["status"] == 200)
    by_kind_worker = Counter((r["kind"], r["worker_id"]) for r in results if r["status"] == 200)
    failures = [r for r in results if r["status"] != 200]

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

    if failures:
        print(f"\nFailures: {len(failures)}")


if __name__ == "__main__":
    main()
