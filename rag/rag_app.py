from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from rag.prompt_builder import PromptBuilder
from rag.services.qdrant_service import VectorDBService
from rag.services.file_processing_service import PDFProcessingService


def _load_master_urls() -> dict[str, str]:
    """Load allowed masters mapping.

    Expected env var:
      MASTER_URLS='{"master1":"http://master1:7000","master2":"http://master2:7000"}'

    This mapping is also used as an allow-list to prevent SSRF.
    """
    raw = os.getenv("MASTER_URLS", "{}").strip()
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError("MASTER_URLS must be valid JSON") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("MASTER_URLS must be a JSON object of {master_id: url}")

    out: dict[str, str] = {}
    for key, value in parsed.items():
        if isinstance(key, str) and isinstance(value, str) and key and value:
            out[key] = value.rstrip("/")

    return out


@lru_cache(maxsize=1)
def _master_urls_cached() -> dict[str, str]:
    return _load_master_urls()


class RagGenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=64, ge=1, le=512)

    # Retrieval controls
    top_k: int = Field(default=3, ge=0, le=20)
    book_id: int | None = Field(default=None)

    # Optional explicit master routing
    master_id: str | None = Field(default=None)


@lru_cache(maxsize=1)
def _vector_service() -> VectorDBService:
    return VectorDBService()


@lru_cache(maxsize=1)
def _prompt_builder() -> PromptBuilder:
    return PromptBuilder(_vector_service())


@lru_cache(maxsize=1)
def _pdf_processor() -> PDFProcessingService:
    return PDFProcessingService()


def _pick_master_url(master_id: str | None, x_master_id: str | None) -> tuple[str, str]:
    master_urls = _master_urls_cached()
    if not master_urls:
        raise HTTPException(
            status_code=500,
            detail="RAG is not configured with any masters (set MASTER_URLS)",
        )

    selected = master_id or x_master_id
    if selected:
        if selected not in master_urls:
            raise HTTPException(status_code=400, detail=f"unknown master_id: {selected}")
        return selected, master_urls[selected]

    # Fallback: pick the first master deterministically.
    # (If you want real load balancing, have NGINX send X-Master-Id.)
    first_id = next(iter(master_urls.keys()))
    return first_id, master_urls[first_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    connect_timeout = float(os.getenv("RAG_CONNECT_TIMEOUT_SEC", "2"))
    read_timeout = float(os.getenv("RAG_READ_TIMEOUT_SEC", "800"))
    write_timeout = float(os.getenv("RAG_WRITE_TIMEOUT_SEC", "60"))
    pool_timeout = float(os.getenv("RAG_POOL_TIMEOUT_SEC", "2"))

    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
            pool=pool_timeout,
        )
    )

    # Warm up heavy components at startup for throughput.
    _ = _prompt_builder()
    _ = _master_urls_cached()

    yield

    await app.state.http.aclose()


app = FastAPI(
    title="RAG Service",
    lifespan=lifespan,
    root_path=os.getenv("RAG_ROOT_PATH", ""),
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.post("/generate")
async def generate(
    payload: RagGenerateRequest,
    request: Request,
    x_master_id: str | None = Header(default=None, alias="X-Master-Id"),
    x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
) -> dict[str, Any]:
    # 1) Enhance prompt via retrieval
    enhanced = _prompt_builder().build_prompt(
        question=payload.prompt,
        book_id=payload.book_id,
        top_k=payload.top_k,
    )

    # 2) Decide where to forward (allow-list)
    selected_master_id, master_url = _pick_master_url(payload.master_id, x_master_id)

    # 3) Forward to master /generate
    http: httpx.AsyncClient = request.app.state.http
    headers = {}
    if x_request_id:
        headers["X-Request-Id"] = x_request_id

    try:
        resp = await http.post(
            f"{master_url}/generate",
            json={"prompt": enhanced, "max_new_tokens": payload.max_new_tokens},
            headers=headers,
        )
        # Preserve master error payloads verbatim.
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.json() if resp.content else resp.text)
        data = resp.json()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail=f"master timeout (master_id={selected_master_id})") from exc
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=502, detail=f"master connect failed (master_id={selected_master_id})") from exc

    return data


@app.post("/documents/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    book_id: int = Form(...),
    chunk_size: int = Form(default=1200),
    chunk_overlap: int = Form(default=200),
    min_chunk_chars: int = Form(default=50),
    embedding_batch_size: int = Form(default=32),
) -> dict[str, Any]:
    """Ingest a PDF into the vector DB.

    - Extracts text per page
    - Chunks text into overlapping windows
    - Embeds chunks
    - Upserts into Qdrant with payload: book_id, page_number, chunk_index, text, filename
    """

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="file must be a .pdf")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    processor = _pdf_processor()
    vector_service = _vector_service()

    try:
        pages = await run_in_threadpool(processor.extract_pages, pdf_bytes)
        chunks = await run_in_threadpool(
            processor.chunk_pages,
            pages,
            chunk_size,
            chunk_overlap,
            min_chunk_chars,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"failed to process pdf: {exc}") from exc

    if not chunks:
        return {
            "status": "ok",
            "book_id": book_id,
            "filename": file.filename,
            "pages_extracted": len(pages),
            "chunks": 0,
            "inserted": 0,
        }

    texts = [c.text for c in chunks]

    try:
        vectors = await run_in_threadpool(
            vector_service.embedding_service.get_embeddings,
            texts,
            embedding_batch_size,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"embedding failed: {exc}") from exc

    payloads: list[dict[str, Any]] = []
    for c in chunks:
        payloads.append(
            {
                "book_id": book_id,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "filename": file.filename,
            }
        )

    try:
        await run_in_threadpool(vector_service.insert_vectors, vectors, payloads)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"vector db insert failed: {exc}") from exc

    return {
        "status": "ok",
        "book_id": book_id,
        "filename": file.filename,
        "pages_extracted": len(pages),
        "chunks": len(chunks),
        "inserted": len(vectors),
    }
