from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Iterable

from pypdf import PdfReader


@dataclass(frozen=True)
class TextChunk:
    text: str
    page_number: int | None = None
    chunk_index: int | None = None


_whitespace_re = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u00a0", " ")
    text = _whitespace_re.sub(" ", text)
    return text.strip()


class PDFProcessingService:
    """Extracts text from PDFs and chunks it for embedding.

    This is intentionally simple (page extraction + character chunking) so it is
    stable and predictable.
    """

    def extract_pages(self, pdf_bytes: bytes) -> list[tuple[int, str]]:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages: list[tuple[int, str]] = []
        for idx, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = _normalize_text(raw)
            if cleaned:
                pages.append((idx, cleaned))
        return pages

    def chunk_pages(
        self,
        pages: Iterable[tuple[int, str]],
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        min_chunk_chars: int = 50,
    ) -> list[TextChunk]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")

        chunks: list[TextChunk] = []
        global_index = 0

        for page_number, page_text in pages:
            text = _normalize_text(page_text)
            if not text:
                continue

            start = 0
            while start < len(text):
                end = min(len(text), start + chunk_size)
                piece = text[start:end]
                piece = _normalize_text(piece)

                if len(piece) >= min_chunk_chars:
                    global_index += 1
                    chunks.append(
                        TextChunk(
                            text=piece,
                            page_number=page_number,
                            chunk_index=global_index,
                        )
                    )

                if end >= len(text):
                    break

                start = max(0, end - chunk_overlap)

        return chunks
