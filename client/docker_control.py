"""Thin wrapper around `docker compose` so the test UI can start/stop services.

Runs from the project root (the directory containing docker-compose.yml). All
calls are async via asyncio.create_subprocess_exec to avoid blocking the
FastAPI event loop while docker waits.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"


class DockerError(RuntimeError):
    pass


async def _run(*args: str, timeout: float = 60.0) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(PROJECT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise DockerError(f"`{' '.join(args)}` timed out after {timeout}s")
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


def _parse_compose_services_from_file() -> list[str]:
    """Read service names from docker-compose.yml without needing PyYAML.

    The file uses simple, well-formatted YAML — top-level keys under `services:`
    that are indented exactly two spaces are the service names. Anything fancier
    would warrant PyYAML, but this matches the current file shape.
    """
    if not COMPOSE_FILE.exists():
        raise DockerError(f"docker-compose.yml not found at {COMPOSE_FILE}")
    services: list[str] = []
    in_services = False
    with COMPOSE_FILE.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.startswith("services:"):
                in_services = True
                continue
            if in_services:
                if not line.startswith(" "):
                    break  # left the services block
                if line.startswith("  ") and not line.startswith("   "):
                    name = line.strip().rstrip(":")
                    if name and not name.startswith("#"):
                        services.append(name)
    return services


async def list_services() -> list[dict[str, Any]]:
    """Return all services defined in docker-compose.yml with their current state.

    Status comes from `docker compose ps --format json --all`. Services that
    aren't running yet show up as state=stopped.
    """
    defined = _parse_compose_services_from_file()
    state_by_name: dict[str, dict[str, Any]] = {}

    rc, stdout, stderr = await _run("docker", "compose", "ps", "--all", "--format", "json")
    if rc != 0:
        raise DockerError(f"docker compose ps failed: {stderr.strip()}")

    # Output is either a JSON array or one JSON object per line, depending on
    # docker compose version. Handle both.
    text = stdout.strip()
    if text.startswith("["):
        try:
            entries = json.loads(text)
        except json.JSONDecodeError:
            entries = []
    else:
        entries = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    for entry in entries:
        name = entry.get("Service") or entry.get("Name") or ""
        if name:
            state_by_name[name] = {
                "state": entry.get("State", "unknown"),
                "status": entry.get("Status", ""),
                "container": entry.get("Name", ""),
            }

    return [
        {
            "name": svc,
            "state": state_by_name.get(svc, {}).get("state", "stopped"),
            "status": state_by_name.get(svc, {}).get("status", ""),
            "container": state_by_name.get(svc, {}).get("container", ""),
        }
        for svc in defined
    ]


async def start_service(name: str) -> dict[str, Any]:
    rc, stdout, stderr = await _run("docker", "compose", "up", "-d", name, timeout=180.0)
    if rc != 0:
        raise DockerError(f"failed to start {name}: {stderr.strip() or stdout.strip()}")
    return {"name": name, "action": "start", "ok": True}


async def stop_service(name: str) -> dict[str, Any]:
    rc, stdout, stderr = await _run("docker", "compose", "stop", name, timeout=60.0)
    if rc != 0:
        raise DockerError(f"failed to stop {name}: {stderr.strip() or stdout.strip()}")
    return {"name": name, "action": "stop", "ok": True}


async def start_all() -> dict[str, Any]:
    """Bring up every service defined in docker-compose.yml.

    `docker compose up -d` with no service argument starts all services. Build
    can take a while on first run, hence the longer timeout.
    """
    rc, stdout, stderr = await _run("docker", "compose", "up", "-d", timeout=600.0)
    if rc != 0:
        raise DockerError(f"failed to start all: {stderr.strip() or stdout.strip()}")
    return {"action": "start_all", "ok": True}


async def stop_all() -> dict[str, Any]:
    """Stop every service defined in docker-compose.yml.

    `docker compose stop` with no service argument stops all services. Doesn't
    remove containers — start-all brings them back fast.
    """
    rc, stdout, stderr = await _run("docker", "compose", "stop", timeout=120.0)
    if rc != 0:
        raise DockerError(f"failed to stop all: {stderr.strip() or stdout.strip()}")
    return {"action": "stop_all", "ok": True}


async def docker_available() -> bool:
    try:
        rc, _, _ = await _run("docker", "compose", "version", timeout=10.0)
        return rc == 0
    except Exception:
        return False
