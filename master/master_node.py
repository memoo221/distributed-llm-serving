import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx

from master.services.config import (
    MASTER_QUEUE_MAXSIZE,
    MASTER_REQUEST_DEADLINE_SEC,
    SCHEDULER_MAX_DISPATCH_PER_TICK,
    SCHEDULER_TICK_SEC,
    GPU_ROUTE_THRESHOLD,
    GPU_BUSY_THRESHOLD,
)
from master.services.forwarder import Forwarder
from master.services.models import WorkerState
from master.services.registry import WorkerRegistry
from master.services.scheduler import estimate_request_score


class MasterQueueFull(Exception):
    pass


class MasterRequestTimeout(Exception):
    pass


_RETRY_STATUSES = {429, 500, 502, 503, 504}


@dataclass
class QueuedRequest:
    prompt: str
    max_new_tokens: int
    request_score: int
    future: asyncio.Future
    deadline_at: float
    # Workers we've permanently given up on for THIS request (e.g. they
    # returned a non-retryable HTTP status). Transient transport errors do
    # NOT add to this set — the registry's failure-cooldown is enough to
    # skip a flaky worker briefly without burning it for the request lifetime.
    tried_workers: set[str] = field(default_factory=set)
    # Total dispatch attempts for this request. Bounded to prevent a queue
    # of always-failing requests from looping forever inside the deadline.
    attempts: int = 0


# Hard cap on per-request dispatches. With 4 workers (2 CPU + 2 GPU) and
# transient errors that don't blacklist, this gives every worker a couple
# of chances before we give up.
_MAX_ATTEMPTS_PER_REQUEST = 6


class MasterNode:
    """Per-master queue + periodic scheduler.

    Design goals:
    - Keep code simple and non-blocking.
    - Apply backpressure with a bounded queue.
    - Dispatch requests at a fixed cadence (tick) with a max-per-tick cap.
    - Use local in-flight counters (more real-time than heartbeats).

    This does NOT implement true token-level inference batching; it implements
    batched *scheduling* (dispatching a group of queued requests per tick).
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        forwarder: Forwarder,
        queue_maxsize: int = MASTER_QUEUE_MAXSIZE,
        tick_sec: float = SCHEDULER_TICK_SEC,
        max_dispatch_per_tick: int = SCHEDULER_MAX_DISPATCH_PER_TICK,
        request_deadline_sec: float = MASTER_REQUEST_DEADLINE_SEC,
    ) -> None:
        self._registry = registry
        self._forwarder = forwarder

        self._queue: asyncio.Queue[QueuedRequest] = asyncio.Queue(maxsize=queue_maxsize)
        self._tick_sec = tick_sec
        self._max_dispatch = max_dispatch_per_tick
        self._request_deadline_sec = request_deadline_sec

        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

        # Local in-flight counters per worker_id (tracked by this master).
        self._local_inflight: dict[str, int] = {}

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def queue_size(self) -> int:
        return self._queue.qsize()

    async def generate(self, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        deadline_at = loop.time() + self._request_deadline_sec

        item = QueuedRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            request_score=estimate_request_score(prompt, max_new_tokens),
            future=fut,
            deadline_at=deadline_at,
        )

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            raise MasterQueueFull("master queue is full") from exc

        remaining = max(0.0, deadline_at - loop.time())
        try:
            return await asyncio.wait_for(fut, timeout=remaining)
        except asyncio.TimeoutError as exc:
            # Scheduler might still complete later; cancel to avoid leaking work.
            if not fut.done():
                fut.cancel()
            raise MasterRequestTimeout("timed out in master queue") from exc

    async def _scheduler_loop(self) -> None:
        loop = asyncio.get_running_loop()

        while not self._stopping.is_set():
            await asyncio.sleep(self._tick_sec)

            batch: list[QueuedRequest] = []
            for _ in range(self._max_dispatch):
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                batch.append(item)

            if not batch:
                continue

            now = loop.time()
            for item in batch:
                if item.future.cancelled() or item.future.done():
                    continue
                if now >= item.deadline_at:
                    item.future.set_exception(MasterRequestTimeout("timed out in master queue"))
                    continue

                worker = self._pick_worker_for(item)
                if worker is None:
                    # No capacity right now; put it back for a later tick.
                    # If the queue is full, fail fast instead of silently dropping.
                    try:
                        self._queue.put_nowait(item)
                    except asyncio.QueueFull:
                        item.future.set_exception(MasterQueueFull("master queue is full"))
                    continue

                item.attempts += 1
                self._local_inflight[worker.worker_id] = self._local_inflight.get(worker.worker_id, 0) + 1
                asyncio.create_task(self._dispatch(item, worker))

    def _effective_load_with_local(self, w: WorkerState) -> float:
        local = self._local_inflight.get(w.worker_id, 0)
        return (w.active_requests + w.queue_depth + local) / max(1, w.slots)

    def _pick_worker_for(self, item: QueuedRequest) -> WorkerState | None:
        workers = self._registry.snapshot()

        # Exclude workers we've already tried for this request.
        eligible = [w for w in workers if w.worker_id not in item.tried_workers]

        gpu = [w for w in eligible if w.device_type == "gpu"]
        cpu = [w for w in eligible if w.device_type == "cpu"]

        # Helper: workers with remaining capacity.
        def has_capacity(w: WorkerState) -> bool:
            return self._effective_load_with_local(w) < 1.0

        # Large requests: GPU only.
        if item.request_score >= GPU_ROUTE_THRESHOLD:
            gpu_cap = [w for w in gpu if has_capacity(w)]
            if not gpu_cap:
                return None
            return min(gpu_cap, key=self._effective_load_with_local)

        # Small requests: prefer GPU while not "busy".
        gpu_headroom = [w for w in gpu if self._effective_load_with_local(w) < GPU_BUSY_THRESHOLD]
        if gpu_headroom:
            return min(gpu_headroom, key=self._effective_load_with_local)

        # Otherwise overflow to CPU if it has capacity.
        cpu_cap = [w for w in cpu if has_capacity(w)]
        if cpu_cap:
            # JSQ(2) sample to avoid stampede while keeping it simple.
            if len(cpu_cap) <= 2:
                return min(cpu_cap, key=self._effective_load_with_local)
            import random

            sample = random.sample(cpu_cap, 2)
            return min(sample, key=self._effective_load_with_local)

        # If no CPU capacity, try GPU (even if above busy threshold) but still under saturation.
        gpu_cap = [w for w in gpu if has_capacity(w)]
        if gpu_cap:
            return min(gpu_cap, key=self._effective_load_with_local)

        return None

    async def _dispatch(self, item: QueuedRequest, worker: WorkerState) -> None:
        try:
            result = await self._forwarder.forward_to_worker(
                worker,
                prompt=item.prompt,
                max_new_tokens=item.max_new_tokens,
            )
            if not item.future.cancelled() and not item.future.done():
                item.future.set_result(result)

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in _RETRY_STATUSES:
                self._registry.mark_failed(worker.worker_id)
                item.tried_workers.add(worker.worker_id)
                await self._retry_or_fail(item)
            else:
                if not item.future.cancelled() and not item.future.done():
                    item.future.set_exception(exc)

        except (
            # Anything transport-level is transient by assumption — retry.
            # httpx.TransportError covers TimeoutException, ConnectError,
            # ReadError, WriteError, RemoteProtocolError, PoolTimeout,
            # NetworkError. These show up when the master <-> Thunder path
            # hiccups mid-generation (cloudflared / tnr edge resets, brief
            # NAT rebinds, ~3s tunnel lag). We do NOT add the worker to
            # tried_workers here — the registry cooldown briefly skips it,
            # and it becomes eligible again automatically. With only 2 GPU
            # workers, permanent blacklisting on every transport hiccup
            # exhausts the pool fast and forces the request into a long
            # queue wait that ends in MasterRequestTimeout.
            httpx.TransportError,
        ) as exc:
            self._registry.mark_failed(worker.worker_id)
            await self._retry_or_fail(item)

        except Exception as exc:
            # Last-resort: don't surface a raw httpx/json exception to the
            # client (FastAPI would render it as 500). Convert to AllRetriesFailed
            # so master_router maps it to a clean 502 with the worker list.
            from master.services.forwarder import AllRetriesFailed
            if not item.future.cancelled() and not item.future.done():
                tried = list(item.tried_workers) + [worker.worker_id]
                item.future.set_exception(AllRetriesFailed(tried))

        finally:
            self._local_inflight[worker.worker_id] = max(
                0, self._local_inflight.get(worker.worker_id, 1) - 1
            )

    async def _retry_or_fail(self, item: QueuedRequest) -> None:
        loop = asyncio.get_running_loop()
        if item.future.cancelled() or item.future.done():
            return
        if loop.time() >= item.deadline_at:
            item.future.set_exception(MasterRequestTimeout("timed out in master queue"))
            return
        if item.attempts >= _MAX_ATTEMPTS_PER_REQUEST:
            from master.services.forwarder import AllRetriesFailed
            item.future.set_exception(AllRetriesFailed(list(item.tried_workers)))
            return

        # Re-enqueue; next tick will pick a different worker (or the same one
        # again once its cooldown expires, for transport-level transients).
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            item.future.set_exception(MasterQueueFull("master queue is full"))
