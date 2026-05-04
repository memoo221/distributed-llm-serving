import json
import os


# How long a worker can be silent before being considered stale (3 missed heartbeats)
STALE_AFTER_SEC: float = float(os.getenv("WORKER_STALE_SEC", "15"))

# Cooldown after a worker returns a transient error
FAILURE_COOLDOWN_SEC: float = float(os.getenv("WORKER_FAILURE_COOLDOWN_SEC", "5"))

# request_score threshold above which the request goes GPU-only
GPU_ROUTE_THRESHOLD: int = int(os.getenv("GPU_ROUTE_THRESHOLD", "256"))

# Fraction of GPU effective_load above which small requests overflow to CPU.
# 0.85 × GPU_SLOTS = spill threshold. Lower = earlier CPU spillover.
GPU_BUSY_THRESHOLD: float = float(os.getenv("GPU_BUSY_THRESHOLD", "0.85"))

# Concurrency capacity of a single GPU worker.
# Groq free tier: 30 RPM ≈ 0.5 req/s; at ~2s avg latency → ~1 slot.
# Set higher (e.g. 8) if using a paid Groq tier or a real local GPU.
GPU_SLOTS: int = int(os.getenv("GPU_SLOTS", "3"))

# API key the master injects on outbound /generate calls to workers
WORKER_API_KEY: str = os.getenv("WORKER_API_KEY", "")

# Forwarder timeouts (seconds)
CONNECT_TIMEOUT: float = float(os.getenv("WORKER_CONNECT_TIMEOUT", "2.0"))
READ_TIMEOUT: float = float(os.getenv("WORKER_READ_TIMEOUT", "60.0"))
WRITE_TIMEOUT: float = float(os.getenv("WORKER_WRITE_TIMEOUT", "5.0"))
POOL_TIMEOUT: float = float(os.getenv("WORKER_POOL_TIMEOUT", "2.0"))

# Retry delays (seconds), jitter applied on top
RETRY_DELAYS: list[float] = [0.1, 0.3]

# How long a request will wait for a suitable worker to free up before 503-ing.
# Workers heartbeat every 5 s, so 30 s = ~6 heartbeat cycles.
QUEUE_WAIT_TIMEOUT: float = float(os.getenv("QUEUE_WAIT_TIMEOUT", "120.0"))

# Static worker list bootstrapped from env so workers are reachable before
# their first heartbeat arrives. JSON list: [{worker_id, url, device_type}]
# Each master must set its own list — pools must not overlap.
def load_bootstrap_workers() -> list[dict]:
    raw = os.getenv("WORKERS_BOOTSTRAP", "")
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"WORKERS_BOOTSTRAP is not valid JSON: {exc}") from exc
