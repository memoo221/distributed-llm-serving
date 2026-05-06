"""FastAPI server for the distributed-llm-serving test UI.

Run on the host (not in a container) so it can shell out to `docker compose`:

    python -m client.app

Then open http://localhost:8050.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from client import docker_control, runner


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="distributed-llm-serving test UI")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/services")
async def get_services() -> dict:
    if not await docker_control.docker_available():
        raise HTTPException(503, "docker compose not available on PATH")
    try:
        services = await docker_control.list_services()
    except docker_control.DockerError as exc:
        raise HTTPException(500, str(exc))
    return {"services": services}


class ServiceAction(BaseModel):
    name: str


@app.post("/api/services/start")
async def post_start(action: ServiceAction) -> dict:
    try:
        return await docker_control.start_service(action.name)
    except docker_control.DockerError as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/services/stop")
async def post_stop(action: ServiceAction) -> dict:
    try:
        return await docker_control.stop_service(action.name)
    except docker_control.DockerError as exc:
        raise HTTPException(500, str(exc))


class RunRequest(BaseModel):
    target_url: str = Field(default="http://localhost:8008/generate")
    total_requests: int = Field(default=100, ge=1, le=100_000)
    concurrency: int = Field(default=20, ge=1, le=2000)
    prompt: str = Field(default="Write a short haiku about distributed systems.")
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    request_timeout_sec: float = Field(default=600.0, ge=1.0, le=3600.0)


@app.post("/api/run")
async def post_run(req: RunRequest) -> dict:
    state = runner.start_run(
        target_url=req.target_url,
        total_requests=req.total_requests,
        concurrency=req.concurrency,
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        request_timeout=req.request_timeout_sec,
    )
    return {"run_id": state.run_id}


@app.get("/api/run/{run_id}")
async def get_run(run_id: str) -> dict:
    state = runner.get_run(run_id)
    if state is None:
        raise HTTPException(404, "run not found")
    return state.snapshot()


@app.get("/api/run/{run_id}/results")
async def get_run_results(run_id: str) -> dict:
    state = runner.get_run(run_id)
    if state is None:
        raise HTTPException(404, "run not found")
    return {
        "run_id": run_id,
        "results": [
            {
                "index": r.index,
                "status_code": r.status_code,
                "latency_sec": round(r.latency_sec, 3),
                "worker_id": r.worker_id,
                "master_id": r.master_id,
                "error": r.error,
            }
            for r in state.results
        ],
    }


@app.post("/api/run/{run_id}/cancel")
async def post_cancel(run_id: str) -> dict:
    ok = await runner.cancel_run(run_id)
    if not ok:
        raise HTTPException(404, "run not found or already finished")
    return {"run_id": run_id, "cancelled": True}


@app.get("/api/runs")
async def get_runs() -> dict:
    return {"runs": runner.list_runs()}


def main() -> None:
    import uvicorn

    uvicorn.run("client.app:app", host="127.0.0.1", port=8050, reload=False)


if __name__ == "__main__":
    main()
