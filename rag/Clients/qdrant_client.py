from __future__ import annotations

import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class QdrantClientManager:
    def __init__(self):
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        self._client = QdrantClient(host=host, port=port)

    @property
    def client(self) -> QdrantClient:
        return self._client

    def create_collection_if_not_exists(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
    ):
        existing = [
            c.name for c in self._client.get_collections().collections
        ]

        if collection_name not in existing:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance,
                ),
            )


