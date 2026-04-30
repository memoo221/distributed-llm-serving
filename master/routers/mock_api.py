import os
import socket
import time
from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field


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


def _sleep_if_requested(delay_ms: int) -> None:
    if delay_ms > 0:
        time.sleep(delay_ms / 1000)


@router.get("/")
def root(
    request: Request,
    delay_ms: int = Query(default=0, ge=0, le=10000),
) -> dict[str, Any]:
    _sleep_if_requested(delay_ms)
    return {
        "message": "request served by mock master",
        **_base_payload(),
        "path": str(request.url.path),
        "delay_ms": delay_ms,
    }


@router.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        **_base_payload(),
    }


_HEARTBEATS: dict[str, dict[str, Any]] = {}


@router.post("/heartbeat")
def heartbeat(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Receive a heartbeat from a worker. Mock implementation: store the latest
    payload per worker_id and log it. Returns ack with current master id.
    """
    worker_id = payload.get("worker_id", "unknown")
    received_at = time.time()
    _HEARTBEATS[worker_id] = {**payload, "received_at": received_at}
    print(
        f"[{MASTER_ID}] heartbeat from worker_id={worker_id} "
        f"device_type={payload.get('device_type')} "
        f"total_requests={payload.get('total_requests')} "
        f"errors={payload.get('total_errors')}"
    )
    return {"status": "ok", "master_id": MASTER_ID, "received_at": received_at}


@router.get("/heartbeat/workers")
def list_workers() -> dict[str, Any]:
    """Inspect the most recent heartbeat seen for each worker."""
    return {
        "master_id": MASTER_ID,
        "workers": _HEARTBEATS,
        "count": len(_HEARTBEATS),
    }


@router.post("/generate")
def generate(payload: GenerateRequest, request: Request) -> dict[str, Any]:
    _sleep_if_requested(payload.delay_ms)
    return {
        "response": f"mock response from {MASTER_ID}",
        "request": payload.model_dump(),
        "request_meta": {
            "path": str(request.url.path),
            "x_forwarded_for": request.headers.get("x-forwarded-for"),
            "x_request_id": request.headers.get("x-request-id"),
            "host": request.headers.get("host"),
        },
        **_base_payload(),
    }
