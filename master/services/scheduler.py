import random

from master.services.config import GPU_ROUTE_THRESHOLD
from master.services.models import WorkerState


def estimate_request_score(prompt: str, max_new_tokens: int) -> int:
    input_tokens_est = max(1, len(prompt) // 4)
    return input_tokens_est + max_new_tokens


def pick_worker(
    workers: list[WorkerState],
    request_score: int,
    threshold: int = GPU_ROUTE_THRESHOLD,
    exclude: set[str] | None = None,
) -> WorkerState | None:
    """
    Token-threshold routing:
      score >= threshold → GPU only (no CPU fallback; CPU would time out)
      score <  threshold → CPU first via JSQ(2), fall back to GPU if every CPU saturated
    """
    exclude = exclude or set()
    live = [w for w in workers if w.worker_id not in exclude and w.effective_load < 1.0]

    gpu_workers = [w for w in live if w.device_type == "gpu"]
    cpu_workers = [w for w in live if w.device_type == "cpu"]

    if request_score >= threshold:
        # Large request: GPU only
        if not gpu_workers:
            return None
        return min(gpu_workers, key=lambda w: w.effective_load)

    # Small request: CPU first via JSQ(2)
    if cpu_workers:
        sample = random.sample(cpu_workers, min(2, len(cpu_workers)))
        return min(sample, key=lambda w: w.effective_load)

    # All CPUs saturated — fall back to GPU
    if gpu_workers:
        return min(gpu_workers, key=lambda w: w.effective_load)

    return None
