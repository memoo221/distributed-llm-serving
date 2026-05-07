"""Async load runner for /generate.

Uses asyncio + a semaphore for concurrency control instead of threads. For
HTTP-bound work this scales to thousands of in-flight requests with a few MB
of memory; the same with OS threads would burn ~1MB stack each and thrash the
kernel scheduler. Same observable behavior (N concurrent requests in flight),
much cheaper.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class RequestResult:
    index: int
    status_code: int | None
    latency_sec: float
    worker_id: str | None = None
    master_id: str | None = None
    error: str | None = None
    # Generated text from the worker. Can be long — UI truncates for display
    # but the full string is kept in memory and exported via CSV.
    response: str | None = None


@dataclass
class RunState:
    run_id: str
    target_url: str
    total_requests: int
    concurrency: int
    prompt: str
    max_new_tokens: int
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    results: list[RequestResult] = field(default_factory=list)
    cancelled: bool = False
    error: str | None = None
    _task: asyncio.Task | None = field(default=None, repr=False)

    def snapshot(self) -> dict[str, Any]:
        elapsed = (self.finished_at or time.time()) - self.started_at
        latencies = [r.latency_sec for r in self.results if r.status_code == 200]
        latencies.sort()
        worker_counts = Counter(r.worker_id for r in self.results if r.worker_id)
        master_counts = Counter(r.master_id for r in self.results if r.master_id)
        status_counts = Counter(r.status_code for r in self.results)

        def pct(p: float) -> float | None:
            if not latencies:
                return None
            idx = min(len(latencies) - 1, int(len(latencies) * p))
            return round(latencies[idx], 3)

        throughput = round(self.completed / elapsed, 2) if elapsed > 0 else 0.0
        return {
            "run_id": self.run_id,
            "target_url": self.target_url,
            "total_requests": self.total_requests,
            "concurrency": self.concurrency,
            "completed": self.completed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "in_flight": max(0, min(self.concurrency, self.total_requests - self.completed)),
            "elapsed_sec": round(elapsed, 2),
            "throughput_rps": throughput,
            "p50_sec": pct(0.50),
            "p95_sec": pct(0.95),
            "p99_sec": pct(0.99),
            "max_sec": round(latencies[-1], 3) if latencies else None,
            "status_counts": {str(k): v for k, v in status_counts.items()},
            "worker_counts": dict(worker_counts),
            "master_counts": dict(master_counts),
            "running": self.finished_at is None and not self.cancelled,
            "cancelled": self.cancelled,
            "finished": self.finished_at is not None,
            "error": self.error,
            # Last 50 results for the live tail in the UI; full results returned
            # only on /run/{id}/results to keep the polling endpoint cheap.
            "recent": [
                {
                    "index": r.index,
                    "status_code": r.status_code,
                    "latency_sec": round(r.latency_sec, 3),
                    "worker_id": r.worker_id,
                    "master_id": r.master_id,
                    "error": r.error,
                    "response": r.response,
                }
                for r in self.results[-50:]
            ],
        }


# In-memory run registry. The client app is single-process so this is fine;
# if the user ever wants multi-replica history, swap to a sqlite or file store.
_runs: dict[str, RunState] = {}


def get_run(run_id: str) -> RunState | None:
    return _runs.get(run_id)


def list_runs() -> list[dict[str, Any]]:
    return [
        {
            "run_id": s.run_id,
            "started_at": s.started_at,
            "finished_at": s.finished_at,
            "total_requests": s.total_requests,
            "succeeded": s.succeeded,
            "failed": s.failed,
            "running": s.finished_at is None and not s.cancelled,
        }
        for s in sorted(_runs.values(), key=lambda r: r.started_at, reverse=True)
    ]


async def _fire_one(
    client: httpx.AsyncClient,
    target_url: str,
    prompt: str,
    max_new_tokens: int,
    index: int,
) -> RequestResult:
    start = time.perf_counter()
    try:
        resp = await client.post(
            target_url,
            json={"prompt": prompt, "max_new_tokens": max_new_tokens},
        )
        latency = time.perf_counter() - start
        if resp.status_code == 200:
            data = resp.json()
            return RequestResult(
                index=index,
                status_code=200,
                latency_sec=latency,
                worker_id=data.get("worker_id"),
                master_id=data.get("master_id"),
                response=data.get("response"),
            )
        body = resp.text[:200]
        return RequestResult(
            index=index,
            status_code=resp.status_code,
            latency_sec=latency,
            error=body,
        )
    except httpx.HTTPError as exc:
        return RequestResult(
            index=index,
            status_code=None,
            latency_sec=time.perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
        )


async def _run_load_test(state: RunState, request_timeout: float) -> None:
    limits = httpx.Limits(
        max_connections=state.concurrency,
        max_keepalive_connections=state.concurrency,
    )
    sem = asyncio.Semaphore(state.concurrency)

    async with httpx.AsyncClient(timeout=request_timeout, limits=limits) as client:
        async def guarded(i: int) -> None:
            async with sem:
                if state.cancelled:
                    return
                result = await _fire_one(
                    client, state.target_url, state.prompt, state.max_new_tokens, i
                )
                state.results.append(result)
                state.completed += 1
                if result.status_code == 200:
                    state.succeeded += 1
                else:
                    state.failed += 1

        try:
            await asyncio.gather(*(guarded(i) for i in range(1, state.total_requests + 1)))
        except asyncio.CancelledError:
            state.cancelled = True
            raise

    state.finished_at = time.time()


def start_run(
    target_url: str,
    total_requests: int,
    concurrency: int,
    prompt: str,
    max_new_tokens: int,
    request_timeout: float = 600.0,
) -> RunState:
    run_id = uuid.uuid4().hex[:12]
    state = RunState(
        run_id=run_id,
        target_url=target_url,
        total_requests=total_requests,
        concurrency=concurrency,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    _runs[run_id] = state

    async def _wrapper() -> None:
        try:
            await _run_load_test(state, request_timeout)
        except asyncio.CancelledError:
            state.cancelled = True
            state.finished_at = time.time()
        except Exception as exc:
            state.error = f"{type(exc).__name__}: {exc}"
            state.finished_at = time.time()

    state._task = asyncio.create_task(_wrapper())
    return state


async def cancel_run(run_id: str) -> bool:
    state = _runs.get(run_id)
    if not state or state._task is None or state._task.done():
        return False
    state.cancelled = True
    state._task.cancel()
    try:
        await state._task
    except (asyncio.CancelledError, Exception):
        pass
    return True
