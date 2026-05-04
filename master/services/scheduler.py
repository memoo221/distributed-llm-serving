import random

from master.services.config import GPU_ROUTE_THRESHOLD, GPU_BUSY_THRESHOLD
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
    GPU-first routing with CPU overflow:

      score >= threshold  → GPU only (no CPU fallback; CPU would time out)
      score <  threshold  → GPU first while effective_load < GPU_BUSY_THRESHOLD,
                            overflow to CPU pool via JSQ(2) when GPU is busy,
                            503 only when both tiers are fully saturated.
    """
    exclude = exclude or set()
    live = [w for w in workers if w.worker_id not in exclude and w.effective_load < 1.0]

    gpu_workers = [w for w in live if w.device_type == "gpu"]
    cpu_workers = [w for w in live if w.device_type == "cpu"]

    # Large requests: GPU only, no CPU fallback
    if request_score >= threshold:
        if not gpu_workers:
            return None
        return min(gpu_workers, key=lambda w: w.effective_load)

    # Small requests: GPU first while it has headroom
    available_gpu = [w for w in gpu_workers if w.effective_load < GPU_BUSY_THRESHOLD]
    if available_gpu:
        return min(available_gpu, key=lambda w: w.effective_load)

    # GPU at/above busy threshold: overflow to CPU via JSQ(2)
    if cpu_workers:
        sample = random.sample(cpu_workers, min(2, len(cpu_workers)))
        return min(sample, key=lambda w: w.effective_load)

    # No CPU workers: accept on GPU even above threshold (beats 503)
    if gpu_workers:
        return min(gpu_workers, key=lambda w: w.effective_load)

    return None
