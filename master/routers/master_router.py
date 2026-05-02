import os
import socket
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from master.services.forwarder import AllRetriesFailed, NoWorkerAvailable


MASTER_ID = os.getenv("MASTER_ID", "master-unknown")
START_TIME = time.time()

router = APIRouter()


class GenerateRequest(BaseModel):
    prompt: str = Field(default="ping")
    max_new_tokens: int = Field(default=32, ge=1, le=512)
    delay_ms: int = Field(default=0, ge=0, le=10000)


def _base_payload() -> dict[str, Any]:
    return {
        "master_id": MASTER_ID,
        "hostname": socket.gethostname(),
        "uptime_seconds": round(time.time() - START_TIME, 2),
    }


@router.get("/")
async def root(
    request: Request,
    delay_ms: int = Query(default=0, ge=0, le=10000),
) -> dict[str, Any]:
    return {
        "message": "request served by master",
        **_base_payload(),
        "path": str(request.url.path),
        "delay_ms": delay_ms,
    }


@router.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", **_base_payload()}


@router.post("/heartbeat")
async def heartbeat(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    await request.app.state.registry.update(payload)
    return {"status": "ok", "master_id": MASTER_ID}


@router.get("/heartbeat/workers")
async def list_workers(request: Request) -> dict[str, Any]:
    registry = request.app.state.registry
    workers = registry.all()
    return {
        "master_id": MASTER_ID,
        "workers": {w.worker_id: w.raw for w in workers},
        "count": len(workers),
    }


@router.post("/generate")
async def generate(payload: GenerateRequest, request: Request) -> dict[str, Any]:
    registry = request.app.state.registry
    forwarder = request.app.state.forwarder

    try:
        result = await forwarder.forward_generate(
            registry,
            payload.prompt,
            payload.max_new_tokens,
        )
    except NoWorkerAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except AllRetriesFailed as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": "all_workers_failed", "tried": exc.tried},
        )

    return {
        "response": result.get("answer") or result.get("response"),
        "worker_id": result.get("worker_id"),
        **_base_payload(),
    }


@router.get("/scheduler/workers")
async def scheduler_debug(request: Request) -> dict[str, Any]:
    """Admin endpoint: live scheduler view of all workers."""
    import time as _time
    registry = request.app.state.registry
    now = _time.monotonic()
    workers = registry.all()
    return {
        "master_id": MASTER_ID,
        "workers": [
            {
                "worker_id": w.worker_id,
                "url": w.url,
                "device_type": w.device_type,
                "active_requests": w.active_requests,
                "queue_depth": w.queue_depth,
                "effective_load": round(w.effective_load, 3),
                "slots": w.slots,
                "last_seen_sec_ago": round(now - w.last_seen_monotonic, 1),
                "in_cooldown": now < w.failure_cooldown_until,
            }
            for w in workers
        ],
    }
