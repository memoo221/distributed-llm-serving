import asyncio
import time

from master.services.config import (
    STALE_AFTER_SEC,
    FAILURE_COOLDOWN_SEC,
    MAX_FAILURE_COOLDOWN_SEC,
)
from master.services.models import WorkerState


class WorkerRegistry:
    def __init__(
        self,
        stale_after_sec: float = STALE_AFTER_SEC,
        static_workers: list[dict] | None = None,
    ) -> None:
        self._stale_after = stale_after_sec
        self._workers: dict[str, WorkerState] = {}
        self._lock = asyncio.Lock()
        # Notified on every heartbeat so waiting requests can re-check availability.
        self.worker_available = asyncio.Condition()

        # Trust bootstrap workers as live at startup. If a worker isn't actually
        # up, the first request will time out, mark_failed will put it on
        # cooldown, and traffic shifts to siblings. This avoids a multi-second
        # 503/queue-wait stall on every fresh master before the first heartbeat.
        bootstrap_now = time.monotonic()
        for entry in static_workers or []:
            wid = entry["worker_id"]
            self._workers[wid] = WorkerState(
                worker_id=wid,
                url=entry["url"],
                device_type=entry.get("device_type", "cpu"),
                active_requests=0,
                queue_depth=0,
                gpu_util_pct=None,
                last_seen_monotonic=bootstrap_now,
                raw=entry,
            )

    async def update(self, payload: dict) -> None:
        wid = payload.get("worker_id")
        if not wid:
            return

        device_type = payload.get("device_type") or (
            "gpu" if payload.get("gpu_util_pct") is not None else "cpu"
        )

        async with self._lock:
            existing = self._workers.get(wid)
            state = WorkerState(
                worker_id=wid,
                url=payload.get("url") or (existing.url if existing else ""),
                device_type=device_type,
                active_requests=int(payload.get("active_requests", 0)),
                queue_depth=int(payload.get("queue_depth", 0)),
                gpu_util_pct=payload.get("gpu_util_pct"),
                last_seen_monotonic=time.monotonic(),
                # Preserve any active cooldown and the failure streak across
                # heartbeats — only mark_success resets the streak.
                failure_cooldown_until=(
                    existing.failure_cooldown_until if existing else 0.0
                ),
                consecutive_failures=(
                    existing.consecutive_failures if existing else 0
                ),
                raw=payload,
            )
            self._workers[wid] = state

        async with self.worker_available:
            self.worker_available.notify_all()

    def snapshot(self) -> list[WorkerState]:
        """Return only workers that are live and not in failure cooldown."""
        now = time.monotonic()
        return [
            w for w in self._workers.values()
            if (now - w.last_seen_monotonic) <= self._stale_after
            and now >= w.failure_cooldown_until
        ]

    def all(self) -> list[WorkerState]:
        """All workers regardless of liveness — for admin/debug endpoints."""
        return list(self._workers.values())

    def mark_failed(self, worker_id: str) -> None:
        """Apply exponential cooldown: 5s → 10s → 20s → 40s → 60s cap.

        Without this, a rate-limited worker is repeatedly thrown back into
        rotation after 5s only to be hit by the next concurrent request and
        marked failed again — it never gets a quiet window to recover. The
        backoff lets sibling workers absorb load while the hot worker cools.
        """
        worker = self._workers.get(worker_id)
        if not worker:
            return
        worker.consecutive_failures += 1
        cooldown = min(
            FAILURE_COOLDOWN_SEC * (2 ** (worker.consecutive_failures - 1)),
            MAX_FAILURE_COOLDOWN_SEC,
        )
        worker.failure_cooldown_until = time.monotonic() + cooldown

    def mark_success(self, worker_id: str) -> None:
        """Reset the failure streak after a successful request through this
        worker. Heartbeats alone don't reset it — only proven request success."""
        worker = self._workers.get(worker_id)
        if worker:
            worker.consecutive_failures = 0
