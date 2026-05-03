import asyncio
import random

import httpx

from master.services.config import (
    CONNECT_TIMEOUT,
    READ_TIMEOUT,
    WRITE_TIMEOUT,
    POOL_TIMEOUT,
    RETRY_DELAYS,
    WORKER_API_KEY,
    GPU_ROUTE_THRESHOLD,
    QUEUE_WAIT_TIMEOUT,
)
from master.services.registry import WorkerRegistry
from master.services.scheduler import estimate_request_score, pick_worker


class NoWorkerAvailable(Exception):
    pass


class AllRetriesFailed(Exception):
    def __init__(self, tried: list[str]) -> None:
        self.tried = tried
        super().__init__(f"tried workers: {tried}")


# HTTP status codes that warrant a retry on a different worker
_RETRY_STATUSES = {502, 503, 504}


class Forwarder:
    def __init__(
        self,
        api_key: str = WORKER_API_KEY,
        connect_timeout: float = CONNECT_TIMEOUT,
        read_timeout: float = READ_TIMEOUT,
        write_timeout: float = WRITE_TIMEOUT,
        pool_timeout: float = POOL_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=write_timeout,
                pool=pool_timeout,
            )
        )

    async def _wait_for_worker(
        self,
        registry: WorkerRegistry,
        request_score: int,
        exclude: set[str],
    ) -> None:
        """
        Block until pick_worker finds a candidate or QUEUE_WAIT_TIMEOUT expires.

        Uses registry.worker_available (an asyncio.Condition) which is notified
        on every heartbeat. We also wake periodically (FAILURE_COOLDOWN_SEC) so
        expiring cooldowns are caught even when no new heartbeat arrives.
        asyncio.Condition.wait() must be called while the lock is held; we use
        wait_for on a wrapper coroutine that acquires + waits inside the lock.
        """
        from master.services.config import FAILURE_COOLDOWN_SEC

        loop = asyncio.get_event_loop()
        deadline = loop.time() + QUEUE_WAIT_TIMEOUT

        while True:
            workers = registry.snapshot()
            if pick_worker(workers, request_score, exclude=exclude) is not None:
                return
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise NoWorkerAvailable(
                    f"timed out waiting for a worker (score={request_score})"
                )
            # Wake on the next heartbeat notification, or after cooldown window.
            wait_sec = min(remaining, FAILURE_COOLDOWN_SEC)
            try:
                async with registry.worker_available:
                    await asyncio.wait_for(
                        registry.worker_available.wait(),
                        timeout=wait_sec,
                    )
            except asyncio.TimeoutError:
                pass  # re-check at top of loop

    async def forward_generate(
        self,
        registry: WorkerRegistry,
        prompt: str,
        max_new_tokens: int,
    ) -> dict:
        request_score = estimate_request_score(prompt, max_new_tokens)
        headers = {"X-API-Key": self._api_key} if self._api_key else {}

        tried: list[str] = []
        exclude: set[str] = set()

        # Up to 3 attempts (initial + 2 retries), each on a different worker
        max_attempts = 1 + len(RETRY_DELAYS)
        for attempt in range(max_attempts):
            workers = registry.snapshot()
            worker = pick_worker(workers, request_score, exclude=exclude)

            if worker is None:
                if tried:
                    # Already dispatched at least once — don't queue-wait again,
                    # just fail fast so the retry budget isn't silently eaten.
                    raise AllRetriesFailed(tried)
                # First pick failed: sleep until a heartbeat frees a worker.
                await self._wait_for_worker(registry, request_score, exclude)
                workers = registry.snapshot()
                worker = pick_worker(workers, request_score, exclude=exclude)
                if worker is None:
                    raise NoWorkerAvailable(
                        f"no eligible worker after wait (score={request_score})"
                    )

            tried.append(worker.worker_id)
            exclude.add(worker.worker_id)

            # GPU workers run groq_worker.py which expects {"prompt": ...};
            # CPU workers run worker_router.py which expects {"question": ...}.
            if worker.device_type == "gpu":
                payload = {"prompt": prompt, "max_new_tokens": max_new_tokens}
            else:
                payload = {"question": prompt, "max_new_tokens": max_new_tokens}

            try:
                resp = await self._client.post(
                    f"{worker.url}/generate",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    data["worker_id"] = worker.worker_id
                    return data

                if resp.status_code in _RETRY_STATUSES:
                    registry.mark_failed(worker.worker_id)
                    if attempt < max_attempts - 1:
                        delay = RETRY_DELAYS[attempt]
                        jitter = delay * 0.25 * (random.random() * 2 - 1)
                        await asyncio.sleep(delay + jitter)
                    continue

                # 4xx or other non-retryable — surface immediately
                resp.raise_for_status()

            except httpx.TimeoutException:
                registry.mark_failed(worker.worker_id)
                if attempt < max_attempts - 1:
                    delay = RETRY_DELAYS[attempt]
                    jitter = delay * 0.25 * (random.random() * 2 - 1)
                    await asyncio.sleep(delay + jitter)

            except httpx.ConnectError:
                registry.mark_failed(worker.worker_id)
                if attempt < max_attempts - 1:
                    delay = RETRY_DELAYS[attempt]
                    jitter = delay * 0.25 * (random.random() * 2 - 1)
                    await asyncio.sleep(delay + jitter)

        raise AllRetriesFailed(tried)

    async def close(self) -> None:
        await self._client.aclose()
