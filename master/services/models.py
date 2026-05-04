from dataclasses import dataclass, field


@dataclass
class WorkerState:
    worker_id: str
    url: str
    device_type: str          # "gpu" | "cpu"
    active_requests: int
    queue_depth: int
    gpu_util_pct: float | None
    last_seen_monotonic: float
    failure_cooldown_until: float = 0.0  # monotonic; 0 = not in cooldown
    raw: dict = field(default_factory=dict)

    @property
    def in_flight(self) -> int:
        return self.active_requests + self.queue_depth

    @property
    def slots(self) -> int:
        from master.services.config import GPU_SLOTS
        return GPU_SLOTS if self.device_type == "gpu" else 1

    @property
    def effective_load(self) -> float:
        return self.in_flight / self.slots
