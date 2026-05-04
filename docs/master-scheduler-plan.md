# Master Node Scheduler Plan

## Context

This project is a distributed LLM serving system. The current request flow is:
`Client → NGINX:8008 → Master Nodes:7000 → Worker Nodes`.

NGINX (with `least_conn`) already balances client requests across **2 master replicas** (master1, master2). Each master is currently a stub: it receives heartbeats from workers and stores them in an in-memory dict ([master/routers/mock_api.py:57](../master/routers/mock_api.py#L57)), but its `/generate` endpoint just echoes the request back ([master/routers/mock_api.py:88](../master/routers/mock_api.py#L88)). The **scheduler that decides which worker handles a generation request does not exist yet** — that is what we are building.

### Topology — partitioned per master

```
                    ┌─────────────┐
                    │ NGINX :8008 │
                    └──────┬──────┘
                           │ least_conn
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌──────────┐              ┌──────────┐
        │ master1  │              │ master2  │
        └────┬─────┘              └────┬─────┘
             │ owns                    │ owns
   ┌─────────┴────────┐       ┌────────┴────────┐
   ▼   ▼   ▼   ▼   ▼          ▼   ▼   ▼   ▼   ▼
  GPU CPU CPU CPU CPU        GPU CPU CPU CPU CPU
   (5 workers, m1's pool)     (5 workers, m2's pool)
```

- **Pools are disjoint.** Each master owns exactly **1 GPU + 4 CPU workers**. m1 has no knowledge of m2's workers and vice versa. There is no cross-master routing.
- Workers heartbeat directly to their owning master (set `MASTER_URL=http://master1:7000` on m1's workers, `http://master2:7000` on m2's). **No worker code change** — `MASTER_URL` is already an env var ([workers/kaggle_worker.py:23](../workers/kaggle_worker.py#L23)).

### Worker characteristics

- **GPU workers** (Kaggle Tesla T4, Qwen 2.5 0.5B): ~3.5 req/s sustained, sweet spot **16 concurrent**. 1 per master.
- **CPU workers** (TinyLlama or Qwen): 30–120 s per single request — **30–60× slower** than GPU. 4 per master.
- All workers expose the same contract: `POST /generate` (auth: `X-API-Key`), `GET /health` (no auth), and POST heartbeats every 5 s to `{MASTER_URL}/heartbeat` ([workers/kaggle_worker.py:87-97](../workers/kaggle_worker.py#L87-L97)).
- Heartbeat payload already carries everything needed: `worker_id`, `url` (worker self-URL), `active_requests`, `queue_depth`, `total_requests`, `total_errors`, `gpu_util_pct`, `vram_used_mb`, `timestamp`.
- Workers do **internal dynamic batching** (50 ms window, batch size 8). Raw `active_requests` underestimates true latency impact; the scheduler must look at queue depth too.

### Capacity envelope (per master)

| Tier | Workers | Concurrent | Sustained throughput |
|------|---------|------------|----------------------|
| GPU  | 1       | 16         | ~3.5 req/s            |
| CPU  | 4       | 4          | ~0.2 req/s (combined) |
| **Total** | **5** | **~20** | **~3.7 req/s** |

System total (2 masters): ~40 concurrent, ~7.4 req/s. **CPU tier exists primarily for graceful degradation under burst, not for sustained throughput.**

### Intended outcome

A master-node scheduler that turns the stub into a real router — one that prefers the GPU until it saturates, falls back to the CPU pool under burst, handles worker failures gracefully, and never blocks the FastAPI event loop.

---

## Recommended Algorithm: token-threshold routing with CPU-first for short prompts

The previous design was unconditionally GPU-first. This version adds a **request-size signal**: short prompts go to a CPU first (only falling back to the GPU if no CPU has capacity), while long prompts skip the CPU tier and go straight to the GPU. The intuition: the GPU is 30–60× faster, but its sweet spot is **batching**. Filling its batch window with cheap short prompts wastes the resource that long prompts actually need. Sending short prompts to the CPUs absorbs them on hardware that would otherwise sit idle, and reserves the GPU's capacity for the work that benefits from it.

### The size signal — `request_score`

We need a number per request that estimates how expensive it will be. We don't load a tokenizer at the master (extra dep, extra latency). Instead, approximate:

```
input_tokens_est = max(1, len(prompt) // 4)        # ~4 chars/token for English BPE
output_tokens    = payload.max_new_tokens           # already in the request schema
request_score    = input_tokens_est + output_tokens
```

For an autoregressive model, **decode (output) dominates total time**, so `max_new_tokens` is the strongest predictor. The `len(prompt)/4` term catches the prefill cost on long contexts. Both are O(1) — no model load, no external call.

### The threshold — `GPU_ROUTE_THRESHOLD`

```
GPU_ROUTE_THRESHOLD = 256   # tokens; configurable via env
```

- `request_score < 256` → **try CPU first**, fall back to GPU if no CPU has capacity.
- `request_score >= 256` → **GPU only** (CPU would take 30–120 s and likely time out).

Why 256? It's roughly the point where CPU latency starts to exceed the request-side `read_timeout` we'd want to use, and it leaves headroom for the GPU's batching window to assemble a useful batch from larger requests. **Tunable, not sacred** — bump it down if CPUs start drowning, bump it up if the GPU starves.

### The load metric — `effective_load`

```
slots(w)        = 16 if w.device_type == "gpu" else 1   # GPU sweet spot 16; CPU sweet spot 1
in_flight(w)    = w.active_requests + w.queue_depth
effective_load  = in_flight(w) / slots(w)               # 0.0 = idle, 1.0 = saturated
```

A CPU is "below effective load" when `effective_load < 1.0`. Normalizing by `slots` lets us compare GPU and CPU on equal footing during fallback decisions.

### Per-request decision

```
1. Compute request_score from the payload.
2. Take a snapshot of live workers (last heartbeat ≤ 15 s, not in cooldown).

3. If request_score >= GPU_ROUTE_THRESHOLD:
     a. If the GPU is live and effective_load < 1.0 → send to GPU. Done.
     b. Else → return HTTP 503 (no fallback to CPU; this prompt is too big for CPU).

4. Else (small request):
     a. Among CPUs with effective_load < 1.0, sample 2 at random,
        send to whichever has lower effective_load. Tie-break: random. Done.
     b. If every CPU is saturated → fall back to the GPU if effective_load < 1.0. Done.
     c. If GPU is also saturated/dead → return HTTP 503.
```

Constant time. No global coordination. No Redis.

### Why this design

- **Token-threshold matches what the hardware is actually good at.** A 50-token "what time is it?" runs in <1 s on a CPU; routing it to the GPU just steals a slot that a 1024-token summarization request needed. Inverting the priority for small requests means the CPU tier earns its keep instead of being a fallback that only sees burst overflow.
- **`max_new_tokens + len(prompt)//4`** is cheap and good enough. We're picking a tier, not estimating wall-clock latency to the millisecond. A 10–20% error doesn't change the routing decision.
- **JSQ(2) on the CPU tier** still applies in step 4a: 4 CPUs and burstiness are real, and JSQ(2)'s randomization de-correlates concurrent picks at O(1) cost.
- **Long requests do NOT fall back to CPU.** A 512-token generation on a CPU is 60–120 s — past the `read_timeout`. Better to 503 early than to silently time out 90 s later.
- **Why not simpler alternatives?**
  - *Pure GPU-first* (the previous design): wastes CPU capacity, fills GPU batch windows with trivial work.
  - *Token count from a real tokenizer*: 5–50 ms overhead, extra dependency, ~10% more accurate — not worth it for a tier decision.
  - *Adaptive threshold*: more state, more failure modes; static is fine for a fixed fleet of 2 masters × (1 GPU + 4 CPU).
  - *Pure least-loaded across all workers*: ignores the speed gap; a small request would still pile onto a GPU that a large request needs.

### Worked example — m1's pool: GPU-1, CPU-1..4

**Case A — small request (`request_score = 80`)**

| Worker | type | active | queue | slots | effective_load |
|--------|------|--------|-------|-------|----------------|
| GPU-1  | gpu  | 4      | 2     | 16    | 0.375          |
| CPU-1  | cpu  | 0      | 0     | 1     | **0.000**      |
| CPU-2  | cpu  | 0      | 0     | 1     | **0.000**      |
| CPU-3  | cpu  | 1      | 0     | 1     | 1.000 (filtered) |
| CPU-4  | cpu  | 0      | 0     | 1     | **0.000**      |

`80 < 256` → step 4a. Eligible CPUs: {CPU-1, CPU-2, CPU-4}. Sample 2 → say {CPU-1, CPU-4} → tied → random → **send to CPU-1**. GPU stays free for big requests.

**Case B — small request, all CPUs saturated**

| Worker | type | active | queue | slots | effective_load |
|--------|------|--------|-------|-------|----------------|
| GPU-1  | gpu  | 8      | 0     | 16    | **0.500**      |
| CPU-1  | cpu  | 1      | 0     | 1     | 1.000 (filtered) |
| CPU-2  | cpu  | 1      | 0     | 1     | 1.000 (filtered) |
| CPU-3  | cpu  | 1      | 0     | 1     | 1.000 (filtered) |
| CPU-4  | cpu  | 1      | 0     | 1     | 1.000 (filtered) |

Step 4a empty → step 4b → GPU `effective_load = 0.5` < 1.0 → **send to GPU-1**.

**Case C — large request (`request_score = 600`)**

`600 >= 256` → step 3. GPU `effective_load < 1.0` → **send to GPU-1**. CPUs are not even considered (they'd time out on this prompt).

**Case D — large request, GPU saturated**

GPU `effective_load = 1.0` → step 3b → **HTTP 503**. We do not fall back to CPU for large requests.

---

## Worker Deployment

CPU workers already run as containers via [docker-compose.workers.yml](../docker-compose.workers.yml) (8 services, `cpu_worker_1..8`, ports 9001–9008, all built from `Dockerfile.cpu`). The split into 8 supports the 2 masters × 4 CPUs partitioning: assign workers 1–4 to m1 and 5–8 to m2 by setting `MASTER_URL` per service:

```yaml
# docker-compose.workers.yml — add to each cpu_worker_N's environment:
- MASTER_URL=http://master1:7000   # for cpu_worker_1..4
- MASTER_URL=http://master2:7000   # for cpu_worker_5..8
- WORKER_ID=worker_1               # already set; keep unique per master pool
```

Each master's `WORKERS_BOOTSTRAP` then lists exactly the 1 GPU + 4 CPUs it owns. GPU workers continue to run on Kaggle (not containerized here) and register via heartbeat as before — no change.

The `master` and `nginx` containers in [docker-compose.yml](../docker-compose.yml) and the workers in `docker-compose.workers.yml` need to share a network so masters can reach the worker containers by service name. Easiest path: a single external network referenced from both files (e.g., `networks: { default: { external: { name: distributed-llm } } }`), created once with `docker network create distributed-llm`.

---

## File Layout

```
master/services/
  __init__.py
  models.py          # WorkerState dataclass
  config.py          # static bootstrap list + env (timeouts, API key)
  registry.py        # WorkerRegistry: heartbeat ingestion, liveness, snapshot
  scheduler.py       # pick_worker() — the JSQ(2) algorithm
  forwarder.py       # forward_generate() — async httpx with retries
```

Public surface (signatures only):

```python
# master/services/models.py
@dataclass
class WorkerState:
    worker_id: str
    url: str
    device_type: str          # "gpu" | "cpu"
    active_requests: int
    queue_depth: int
    gpu_util_pct: float | None
    last_seen_monotonic: float
    raw: dict                 # full last heartbeat (for debug endpoint)

# master/services/registry.py
class WorkerRegistry:
    def __init__(self, stale_after_sec: float = 15.0,
                 static_workers: list[dict] | None = None) -> None: ...
    async def update(self, payload: dict) -> None: ...    # called from /heartbeat
    def snapshot(self) -> list[WorkerState]: ...           # only live workers
    def mark_failed(self, worker_id: str, cooldown_sec: float = 5.0) -> None: ...
    def all(self) -> list[WorkerState]: ...                # for /admin endpoints

# master/services/scheduler.py
GPU_ROUTE_THRESHOLD = 256   # env-overridable

def estimate_request_score(prompt: str, max_new_tokens: int) -> int:
    """input_tokens_est + max_new_tokens, with input ~= len(prompt) // 4."""
    ...

def pick_worker(workers: list[WorkerState],
                request_score: int,
                threshold: int = GPU_ROUTE_THRESHOLD,
                exclude: set[str] | None = None) -> WorkerState | None: ...

# master/services/forwarder.py
class Forwarder:
    def __init__(self, api_key: str,
                 connect_timeout: float = 2.0,
                 read_timeout: float = 60.0) -> None: ...
    async def forward_generate(self, registry: WorkerRegistry,
                               prompt: str,
                               max_new_tokens: int) -> dict: ...
        # Computes request_score, calls pick_worker(snapshot, request_score),
        # forwards to {"question": prompt, "max_new_tokens": ...}, retries on a
        # different eligible worker on transient failure.
    async def close(self) -> None: ...
```

`Forwarder` owns a **single shared `httpx.AsyncClient`** (created once in lifespan, closed on shutdown) — never per-request.

---

## Change to `master/routers/mock_api.py:88`

Both `/generate` and `/heartbeat` must become `async def` so the request path is non-blocking. The module-level `_HEARTBEATS` dict goes away — replaced by `app.state.registry`.

```python
@router.post("/generate")
async def generate(payload: GenerateRequest, request: Request):
    registry: WorkerRegistry = request.app.state.registry
    forwarder: Forwarder    = request.app.state.forwarder
    try:
        result = await forwarder.forward_generate(
            registry,
            payload.prompt,
            payload.max_new_tokens,
        )
    except NoWorkerAvailable:
        raise HTTPException(503, "no live workers")
    except AllRetriesFailed as e:
        raise HTTPException(502, f"all workers failed: {e}")
    return {"response": result.get("answer") or result.get("response"),
            "worker_id": result.get("worker_id"),
            **_base_payload()}

@router.post("/heartbeat")
async def heartbeat(payload: dict[str, Any], request: Request):
    await request.app.state.registry.update(payload)
    return {"status": "ok", "master_id": MASTER_ID}
```

Wire `app.state.registry` and `app.state.forwarder` from a FastAPI lifespan in [master/mock_api.py](../master/mock_api.py) (the FastAPI factory).

---

## Registry Design

- **Storage:** `dict[worker_id, WorkerState]` guarded by a **single `asyncio.Lock`** (FastAPI on uvicorn is single-process per replica and async — `threading.Lock` is unnecessary).
- **Time:** always `time.monotonic()`, never wall clock. The worker's own `timestamp` is informational only; `last_seen_monotonic` is set by the master on ingest.
- **Liveness:** `monotonic() - last_seen_monotonic ≤ 15 s` (= 3 missed 5-s heartbeats). Workers in transient failure cooldown (`mark_failed`) are skipped until cooldown expires.
- **`device_type` derivation:** prefer heartbeat's explicit `device_type`; else infer from `gpu_util_pct is not None`. CPU workers may not send GPU keys.
- **Static bootstrap (per master):** `master/services/config.py` reads a `WORKERS_BOOTSTRAP` env var (JSON list `[{worker_id, url, device_type}]`) so CPU workers (which often don't set `SELF_URL`) are reachable from the first request, before their first heartbeat. Heartbeats then refresh those entries. Each master's bootstrap is independent: m1's `WORKERS_BOOTSTRAP` lists its 1 GPU + 4 CPUs; m2's lists its own 5. Pools must not overlap.
- **Eviction:** lazy. Stale entries stay in the dict but are filtered out by `snapshot()`. No periodic prune required for correctness.
- **No active `/health` probe on request path.** Heartbeats every 5 s are already the freshness signal; an extra probe would double request latency for no information gain. Optional one-time probe for never-heartbeated bootstrap entries on cold start.

---

## Failure / Retry Policy

- `httpx.AsyncClient(timeout=httpx.Timeout(connect=2.0, read=60.0, write=5.0, pool=2.0))`.
- **Max 2 retries** (3 attempts total), each on a **different** worker chosen by `pick_worker(exclude=tried_set)`. NGINX already retries 3× across master replicas — don't layer aggressive retries underneath.
- Retry on: connect error, read timeout, HTTP 502/503/504, generic 5xx. **Never on 4xx** (worker rejected the request — bug, not transience).
- Backoff: 100 ms, 300 ms (jittered ±25%).
- On any failure, call `registry.mark_failed(worker_id, cooldown_sec=5.0)`.
- Auth: master reads `WORKER_API_KEY` once at startup; forwarder injects `X-API-Key` on every outbound `/generate`. Master does **not** propagate the client's `X-API-Key` (master is its own trust boundary).
- Final failure → HTTP 502 with `{"error": "all_workers_failed", "tried": [...]}`. NGINX won't retry 502 with the existing config — that's intentional; the client should see it.

---

## Critical Files to Modify

- [master/routers/mock_api.py](../master/routers/mock_api.py) — convert `/generate` and `/heartbeat` to `async def`, delegate to registry + forwarder, drop the in-process `_HEARTBEATS` dict.
- [master/mock_api.py](../master/mock_api.py) — add a FastAPI lifespan that constructs `WorkerRegistry` + `Forwarder`, attaches them to `app.state`, and closes the httpx client on shutdown.
- [master/services/registry.py](../master/services/registry.py) — **new file**, `WorkerRegistry`.
- [master/services/scheduler.py](../master/services/scheduler.py) — **new file**, `pick_worker` (JSQ(2)).
- [master/services/forwarder.py](../master/services/forwarder.py) — **new file**, async retry forwarder.
- [master/services/models.py](../master/services/models.py) — **new file**, `WorkerState` dataclass.
- [master/services/config.py](../master/services/config.py) — **new file**, static bootstrap list + env loading.

Supporting / optional: [tests/test_scheduler.py](../tests/test_scheduler.py) (unit tests for `pick_worker`).

---

## Verification

End-to-end checks against the **2 master × (1 GPU + 4 CPU)** topology. Reuse `tests/simulate_nginx_lb.py` as the harness pattern.

1. **Cold start, no heartbeats** — send 1 request → expect HTTP 503. Confirms staleness filter.
2. **Small-request steady state** — fire 60 requests of `max_new_tokens=64`, short prompts (`request_score ≈ 80`) at 3 rps. Expect traffic to land on the **CPU workers** (4 per master × 2 masters); GPU `total_requests` stays low. This is the inverted-priority case — verifies that small prompts go CPU-first.
3. **Large-request steady state** — fire 30 requests of `max_new_tokens=512`, long prompts (`request_score ≈ 600`) at 1 rps. Expect all traffic on the **GPUs** (~50/50 across m1-GPU and m2-GPU thanks to NGINX `least_conn`); CPU workers see 0 large requests.
4. **Mixed workload** — 80% small (`score < 256`) + 20% large (`score >= 256`) at 4 rps. Small requests should saturate CPUs first; large requests should always go to GPU. Inspect via `GET /heartbeat/workers` on each master and confirm tier separation.
5. **Saturate the CPUs on one master with small requests** — pin all small-request traffic to m1 by bypassing NGINX and bursting 16 concurrent. Expect: m1's 4 CPUs reach `effective_load = 1.0`, then small requests start falling back to m1-GPU (step 4b). m2 is untouched. No 503s.
6. **Saturate the GPU with large requests** — burst 32 concurrent large requests at m1. m1-GPU hits `effective_load = 1.0`. New large requests **must return 503** (no CPU fallback for big prompts). Verify error code and that no large request silently lands on a CPU.
7. **Kill the GPU on m1 mid-test.** Within ≤ 15 s of last heartbeat, m1 stops routing to it. Small requests continue on m1's CPUs; **large requests start returning 503 on m1**. m2's pool is unchanged. m1 does NOT route to m2's GPU.
8. **5xx injection** — flag one CPU on m1 to 503 ~30% of requests. Master retries on a different m1 worker (eligible CPU first; if all CPUs cooked, GPU); failed worker enters 5 s cooldown; client error rate near 0.
9. **Master failover** (NGINX behavior). Kill master1 mid-test. NGINX retries on master2. All 5 of m1's workers are offline from the system's perspective — m2 only serves from its pool. Capacity halves. Client error rate bounded by NGINX retry budget.
10. **Bootstrap independence** — boot m1 with `WORKERS_BOOTSTRAP` listing only m1's pool, m2 with only m2's. Confirm `GET /heartbeat/workers` on each master never shows the other's workers.
11. **Latency sanity** — p50 GPU forward < 1.5 s, p95 < 4 s. CPU forward p50 < 30 s for `max_new_tokens=64`. `pick_worker` itself < 5 ms.

---

## Out of Scope (do NOT build now)

- **Cross-master routing.** Pools are partitioned by design — m1 only routes to m1's pool. No worker stealing across masters, no Redis/etcd/gossip.
- **Active `/health` probing on the request path.** Heartbeats are the signal.
- **Model-aware routing** (one prompt → one model). Single model contract today.
- **Priority queues / SLO classes / preemption.**
- **Streaming / SSE responses.**
- **Sticky sessions / KV-cache affinity.** Not supported by the worker today.
- **Prometheus metrics export.** Add a debug `GET /scheduler/workers` admin endpoint, skip the full pipeline.
- **Auto-scaling.** Fixed fleet (2 masters × (1 GPU + 4 CPU)).
- **Changes to worker heartbeat schema.** Hard constraint — everything above works with the existing payload.
