from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from typing import Any

import httpx


async def fetch_one(client: httpx.AsyncClient, url: str, index: int) -> dict[str, Any]:
    try:
        response = await client.get(url)
        payload = response.json() if "application/json" in response.headers.get("content-type", "") else {}
        return {
            "index": index,
            "status_code": response.status_code,
            "master_id": payload.get("master_id"),
            "hostname": payload.get("hostname"),
            "delay_ms": payload.get("delay_ms", 0),
            "error_body": None if response.status_code < 400 else response.text,
        }
    except httpx.HTTPError as exc:
        return {
            "index": index,
            "status_code": None,
            "master_id": None,
            "hostname": None,
            "delay_ms": 0,
            "error_body": str(exc),
        }


async def run(base_url: str, requests: int, concurrency: int, delay_ms: int) -> list[dict[str, Any]]:
    url = f"{base_url}/?delay_ms={delay_ms}"
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(timeout=15.0, limits=limits) as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def guarded_fetch(index: int) -> dict[str, Any]:
            async with semaphore:
                return await fetch_one(client, url, index)

        tasks = [guarded_fetch(index) for index in range(1, requests + 1)]
        return await asyncio.gather(*tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate traffic through NGINX and count which master answered.")
    parser.add_argument("--base-url", default="http://localhost:8008", help="NGINX base URL")
    parser.add_argument("--requests", type=int, default=20, help="Total number of requests to send")
    parser.add_argument("--concurrency", type=int, default=8, help="How many requests to run at once")
    parser.add_argument("--delay-ms", type=int, default=1000, help="Artificial backend delay to create overlap")
    args = parser.parse_args()

    results = asyncio.run(
        run(
            base_url=args.base_url.rstrip("/"),
            requests=args.requests,
            concurrency=args.concurrency,
            delay_ms=args.delay_ms,
        )
    )

    counts = Counter(result["master_id"] for result in results if result["master_id"])
    failures = [result for result in results if result["status_code"] != 200]

    print("Per-request results:")
    for result in results:
        if result["status_code"] == 200:
            print(
                f"request={result['index']:02d} "
                f"status=200 "
                f"master={result['master_id']} "
                f"hostname={result['hostname']} "
                f"delay_ms={result['delay_ms']}"
            )
        else:
            print(
                f"request={result['index']:02d} "
                f"status={result['status_code']} "
                f"error={result['error_body']}"
            )

    print("\nDistribution summary:")
    for master_id, count in sorted(counts.items()):
        print(f"{master_id}: {count}")

    if failures:
        print(f"\nFailures: {len(failures)}")


if __name__ == "__main__":
    main()
