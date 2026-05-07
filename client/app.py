"""FastAPI server for the distributed-llm-serving test UI.

Run on the host (not in a container) so it can shell out to `docker compose`:

    python -m client.app

Then open http://localhost:8050.
"""
from __future__ import annotations

from pathlib import Path

import asyncio
import csv
import io

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from client import docker_control, runner


# Masters the UI queries directly for live worker registry data. This is what
# surfaces remote workers (e.g. Thunder GPU instances) that aren't part of
# docker-compose — they're invisible to /api/services but show up here as
# soon as they heartbeat to a master.
MASTER_URLS: list[tuple[str, str]] = [
    ("master1", "http://localhost:7001"),
    ("master2", "http://localhost:7002"),
]


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


@app.post("/api/services/start-all")
async def post_start_all() -> dict:
    try:
        return await docker_control.start_all()
    except docker_control.DockerError as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/workers")
async def get_workers() -> dict:
    """Live workers from each master's registry.

    Surfaces anything that's heartbeating to a master, including non-docker
    workers (e.g. Thunder GPUs). One query per master, in parallel; if a
    master is unreachable we still return whatever the others gave us.
    """
    async def _fetch(master_id: str, base_url: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=2.0) as http:
                resp = await http.get(f"{base_url}/scheduler/workers")
                if resp.status_code != 200:
                    return {"master_id": master_id, "reachable": False, "error": f"HTTP {resp.status_code}", "workers": []}
                data = resp.json()
                return {
                    "master_id": master_id,
                    "reachable": True,
                    "workers": data.get("workers", []),
                }
        except Exception as exc:
            return {"master_id": master_id, "reachable": False, "error": f"{type(exc).__name__}: {exc}", "workers": []}

    masters = await asyncio.gather(*(_fetch(mid, url) for mid, url in MASTER_URLS))
    return {"masters": masters}


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


@app.get("/api/run/{run_id}/csv")
async def get_run_csv(run_id: str) -> Response:
    state = runner.get_run(run_id)
    if state is None:
        raise HTTPException(404, "run not found")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["index", "status_code", "latency_sec", "worker_id", "master_id", "error"])
    for r in state.results:
        writer.writerow([
            r.index,
            r.status_code if r.status_code is not None else "",
            f"{r.latency_sec:.3f}",
            r.worker_id or "",
            r.master_id or "",
            (r.error or "").replace("\n", " ").replace("\r", " "),
        ])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="run_{run_id}.csv"'},
    )


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
