from __future__ import annotations

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
_CWD = os.getcwd()
_removed_paths: list[tuple[int, str]] = []
for idx in range(len(sys.path) - 1, -1, -1):
    candidate = sys.path[idx]
    normalized = candidate or _CWD
    if normalized in {_REPO_ROOT, _CWD}:
        _removed_paths.append((idx, candidate))
        sys.path.pop(idx)

try:
    import modal
except ImportError:  # pragma: no cover - optional for local FastAPI proxy use
    modal = None
finally:
    for idx, candidate in reversed(_removed_paths):
        sys.path.insert(idx, candidate)

load_dotenv()

WORKER_ID = os.getenv("WORKER_ID", "modal_worker")
SELF_URL = os.getenv("SELF_URL", "")
MASTER_URL = os.getenv("MASTER_URL", "")
HEARTBEAT_SEC = float(os.getenv("HEARTBEAT_INTERVAL", "5"))
WORKER_API_KEY = os.getenv("WORKER_API_KEY", "")

MODAL_WORKER_URL = os.getenv("MODAL_WORKER_URL", "").strip()
MODAL_TIMEOUT_SEC = float(os.getenv("MODAL_TIMEOUT_SEC", "180"))
MODAL_API_KEY = os.getenv("MODAL_API_KEY", "").strip()
MODAL_MODEL_NAME = os.getenv(
    "MODAL_MODEL_NAME",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
)
MODAL_MAX_INPUTS = int(os.getenv("MODAL_MAX_INPUTS", "16"))
MODAL_TARGET_INPUTS = int(os.getenv("MODAL_TARGET_INPUTS", "8"))
MODAL_MIN_CONTAINERS = int(os.getenv("MODAL_MIN_CONTAINERS", "0"))
MODAL_MAX_CONTAINERS = int(os.getenv("MODAL_MAX_CONTAINERS", "16"))
MODAL_BUFFER_CONTAINERS = int(os.getenv("MODAL_BUFFER_CONTAINERS", "1"))
MODAL_SCALEDOWN_WINDOW = int(os.getenv("MODAL_SCALEDOWN_WINDOW", "300"))

active_requests = 0
total_requests = 0
total_errors = 0


def build_modal_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if MODAL_API_KEY:
        headers["Authorization"] = f"Bearer {MODAL_API_KEY}"
    return headers


def normalize_modal_response(payload: Any) -> dict[str, Any]:
    if isinstance(payload, str):
        return {"response": payload}

    if isinstance(payload, dict):
        nested = payload.get("data")
        if nested is not None:
            normalized = normalize_modal_response(nested)
            for key, value in payload.items():
                if key != "data" and key not in normalized:
                    normalized[key] = value
            return normalized

        for key in ("response", "answer", "text", "output", "generated_text"):
            value = payload.get(key)
            if isinstance(value, str):
                normalized = {"response": value}
                for extra_key, extra_value in payload.items():
                    if extra_key != key and extra_key not in normalized:
                        normalized[extra_key] = extra_value
                return normalized

    raise ValueError(f"Unsupported Modal response payload: {type(payload).__name__}")


async def call_modal_endpoint(prompt: str, max_new_tokens: int) -> dict[str, Any]:
    if not MODAL_WORKER_URL:
        raise RuntimeError("MODAL_WORKER_URL is not configured")

    payload = {"prompt": prompt, "max_new_tokens": max_new_tokens}
    headers = build_modal_headers()

    async with httpx.AsyncClient(timeout=MODAL_TIMEOUT_SEC) as client:
        response = await client.post(
            MODAL_WORKER_URL,
            json=payload,
            headers=headers,
        )

        # Backward compatibility for older deployed Modal endpoints that were
        # defined with scalar function args, which FastAPI interprets as query
        # params instead of a JSON body.
        if response.status_code == 422:
            response = await client.post(
                MODAL_WORKER_URL,
                params=payload,
                headers=headers,
            )

    if response.status_code == 429:
        raise HTTPException(status_code=429, detail="modal rate limit exceeded")

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        data = response.json()
    else:
        data = response.text

    try:
        return normalize_modal_response(data)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


async def heartbeat_loop() -> None:
    if not MASTER_URL:
        print(f"[{WORKER_ID}] MASTER_URL not set, skipping heartbeats")
        return

    headers = {"X-API-Key": WORKER_API_KEY} if WORKER_API_KEY else {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            payload = {
                "worker_id": WORKER_ID,
                "url": SELF_URL,
                "device_type": "gpu",
                "active_requests": active_requests,
                "queue_depth": 0,
                "total_requests": total_requests,
                "total_errors": total_errors,
                "provider": "modal",
                "timestamp": time.time(),
            }
            try:
                await client.post(f"{MASTER_URL}/heartbeat", json=payload, headers=headers)
            except Exception as exc:  # pragma: no cover - network dependent
                print(f"[{WORKER_ID}] heartbeat failed: {type(exc).__name__}: {exc}")
            await asyncio.sleep(HEARTBEAT_SEC)


@asynccontextmanager
async def lifespan(_: FastAPI):
    print(
        f"[{WORKER_ID}] starting; modal_url={MODAL_WORKER_URL or 'missing'}; "
        f"master={MASTER_URL or 'none'}; self_url={SELF_URL or 'none'}"
    )
    task = asyncio.create_task(heartbeat_loop())
    yield
    task.cancel()


worker_app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str | None = None
    question: str | None = None
    max_new_tokens: int = 256


@worker_app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "worker_id": WORKER_ID,
        "active_requests": active_requests,
        "total_requests": total_requests,
        "total_errors": total_errors,
        "provider": "modal",
        "modal_url_configured": bool(MODAL_WORKER_URL),
    }


@worker_app.post("/generate")
async def generate_request(req: GenerateRequest) -> dict[str, Any]:
    global active_requests, total_requests, total_errors

    prompt = (req.prompt or req.question or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    active_requests += 1
    try:
        result = await call_modal_endpoint(prompt, req.max_new_tokens)
        total_requests += 1
        return result
    except HTTPException:
        total_errors += 1
        raise
    except Exception as exc:
        total_errors += 1
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        active_requests -= 1


if modal is not None:
    app = modal.App("qwen2-5-coder-7b-instruct")

    image = (
        modal.Image.debian_slim()
        .pip_install(
            "fastapi[standard]",
            "torch",
            "transformers",
            "accelerate",
            "huggingface_hub",
        )
    )

    hf_secret = modal.Secret.from_name("huggingface-secret")

    @app.cls(
        gpu="A10G",
        image=image,
        secrets=[hf_secret],
        min_containers=MODAL_MIN_CONTAINERS,
        max_containers=MODAL_MAX_CONTAINERS,
        buffer_containers=MODAL_BUFFER_CONTAINERS,
        scaledown_window=MODAL_SCALEDOWN_WINDOW,
    )
    @modal.concurrent(
        max_inputs=MODAL_MAX_INPUTS,
        target_inputs=MODAL_TARGET_INPUTS,
    )
    class QwenCoderModel:
        @modal.enter()
        def load_model(self) -> None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            token = os.environ["HUGGINGFACEHUB_API_TOKEN"]

            self.tokenizer = AutoTokenizer.from_pretrained(
                MODAL_MODEL_NAME,
                token=token,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                MODAL_MODEL_NAME,
                token=token,
                torch_dtype="auto",
                device_map="auto",
            )

        @modal.method()
        def generate(self, prompt: str, max_new_tokens: int = 256) -> dict[str, Any]:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )
            return {
                "response": self.tokenizer.decode(outputs[0], skip_special_tokens=True),
                "modal_task_id": os.getenv("MODAL_TASK_ID"),
                "modal_region": os.getenv("MODAL_REGION"),
                "modal_model": MODAL_MODEL_NAME,
            }

    @app.function(image=image)
    @modal.fastapi_endpoint(method="POST")
    def generate(prompt: str, max_new_tokens: int = 256) -> dict[str, Any]:
        model = QwenCoderModel()
        result = model.generate.remote(prompt, max_new_tokens)
        result["modal_endpoint_task_id"] = os.getenv("MODAL_TASK_ID")
        return result
