from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from pathlib import Path
from typing import Any

import httpx


async def _post_json(client: httpx.AsyncClient, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    r = await client.post(url, json=payload)
    return {
        "status": r.status_code,
        "json": r.json() if r.headers.get("content-type", "").startswith("application/json") else None,
        "text": r.text,
    }


async def _post_pdf_multipart(
    client: httpx.AsyncClient,
    url: str,
    *,
    pdf_path: str,
    book_id: int,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
    embedding_batch_size: int,
) -> dict[str, Any]:
    p = Path(pdf_path)
    pdf_bytes = p.read_bytes()
    files = {"file": (p.name, pdf_bytes, "application/pdf")}
    data = {
        "book_id": str(book_id),
        "chunk_size": str(chunk_size),
        "chunk_overlap": str(chunk_overlap),
        "min_chunk_chars": str(min_chunk_chars),
        "embedding_batch_size": str(embedding_batch_size),
    }

    r = await client.post(url, files=files, data=data)
    return {
        "status": r.status_code,
        "json": r.json() if r.headers.get("content-type", "").startswith("application/json") else None,
        "text": r.text,
    }


async def run(
    base_url: str,
    n: int,
    concurrency: int,
    pdf_path: str | None,
    book_id: int,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
    embedding_batch_size: int,
) -> None:
    base_url = base_url.rstrip("/")

    direct_url = f"{base_url}/generate"
    rag_url = f"{base_url}/rag/generate"
    ingest_url = f"{base_url}/rag/documents/pdf"

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(connect=5.0, read=800.0, write=60.0, pool=5.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # 1) Direct path sanity check
        direct = await _post_json(
            client,
            direct_url,
            {"prompt": "hello", "max_new_tokens": 32},
        )
        print(f"direct: status={direct['status']}")
        if direct["status"] != 200:
            print(direct["text"][:500])
        else:
            j = direct["json"] or {}
            print(f"  master_id={j.get('master_id')} worker_id={j.get('worker_id')}")

        # 1b) Optional: ingest a PDF to Qdrant
        if pdf_path:
            ingest = await _post_pdf_multipart(
                client,
                ingest_url,
                pdf_path=pdf_path,
                book_id=book_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_chars=min_chunk_chars,
                embedding_batch_size=embedding_batch_size,
            )
            print(f"ingest: status={ingest['status']}")
            if ingest["status"] != 200:
                print(ingest["text"][:800])
            else:
                j = ingest["json"] or {}
                print(
                    f"  book_id={j.get('book_id')} pages={j.get('pages_extracted')} chunks={j.get('chunks')} inserted={j.get('inserted')}"
                )

        # 2) RAG path sanity check
        rag = await _post_json(
            client,
            rag_url,
            {"prompt": "What is consistent hashing?", "max_new_tokens": 32, "top_k": 1},
        )
        print(f"rag:    status={rag['status']}")
        if rag["status"] != 200:
            print(rag["text"][:500])
        else:
            j = rag["json"] or {}
            print(f"  master_id={j.get('master_id')} worker_id={j.get('worker_id')}")

        # 3) Distribution check (RAG)
        sem = asyncio.Semaphore(concurrency)

        async def one(i: int) -> tuple[int, str | None, str | None]:
            async with sem:
                res = await _post_json(
                    client,
                    rag_url,
                    {"prompt": f"ping {i}", "max_new_tokens": 16, "top_k": 1},
                )
                if res["status"] != 200:
                    return res["status"], None, None
                j = res["json"] or {}
                return 200, j.get("master_id"), j.get("worker_id")

        results = await asyncio.gather(*(one(i) for i in range(1, n + 1)))

    status_counts = Counter(s for s, _, _ in results)
    master_counts = Counter(m for _, m, _ in results if m)
    worker_counts = Counter(w for _, _, w in results if w)

    print("\nRAG distribution summary:")
    print(f"  statuses: {dict(status_counts)}")
    print(f"  masters:  {dict(master_counts)}")
    print(f"  workers:  {dict(worker_counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test: direct /generate and /rag/generate")
    parser.add_argument("--base-url", default="http://localhost:8008", help="NGINX base URL")
    parser.add_argument("--requests", type=int, default=20, help="How many RAG requests for distribution")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrency for distribution run")
    parser.add_argument("--pdf", dest="pdf_path", default=None, help="Optional: path to a PDF to ingest")
    parser.add_argument("--book-id", type=int, default=2, help="book_id payload filter for ingestion")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--min-chunk-chars", type=int, default=50)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    args = parser.parse_args()

    asyncio.run(
        run(
            args.base_url,
            args.requests,
            args.concurrency,
            args.pdf_path,
            args.book_id,
            args.chunk_size,
            args.chunk_overlap,
            args.min_chunk_chars,
            args.embedding_batch_size,
        )
    )


if __name__ == "__main__":
    main()
