from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List

from rag.services.embedding_service import EmbeddingProvider
from rag.repo.qdrant_repo import QdrantVectorRepository

class VectorDBService:

    def __init__(
        self,
        embedding_service: EmbeddingProvider | None = None,
        vector_repo: QdrantVectorRepository | None = None,
        collection_name: str | None = None,
    ):
        self.embedding_service = embedding_service or EmbeddingProvider(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "dis-content")
        self.vector_repo = vector_repo or QdrantVectorRepository()

        # Ensure collection exists with the correct vector dimension.
        vector_size = getattr(self.embedding_service.model, "get_sentence_embedding_dimension", lambda: None)()
        if not isinstance(vector_size, int) or vector_size <= 0:
            # Fallback: infer from a sample embedding.
            vector_size = len(self.embedding_service.get_embedding("dimension probe"))
        self.vector_repo.ensure_collection_exists(self.collection_name, vector_size=vector_size)
    
    def insert_vectors(
        self,
        vectors: List[List[float]],
        payloads: List[Dict],
    ):
        print(f"Inserting {len(vectors)} vectors into the database...{self.collection_name}") 
        ids = [str(uuid.uuid4()) for _ in vectors]
        self.vector_repo.add_vectors(
            collection_name=self.collection_name,
            vectors=vectors,
            payloads=payloads,
            ids=ids,
        )
        print("Vectors inserted successfully.") 

    def search_vectors(
        self,
        query_vector: List[float],
        book_id: int | None = None,
        top_k: int = 5,
    ):
        filters: Dict[str, Any] | None = {"book_id": book_id} if book_id is not None else None
        results = self.vector_repo.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            filters=filters,
        )

        return [
            {
                "score": r.score,
                "page_number": r.payload.get("page_number"),
                "text": r.payload.get("text"),
            }
            for r in results
        ]
        
    def search_in_book(
        self,
        book_id: int | None,
        question: str,
        top_k: int = 5,
    ):
        query_embedding = self.embedding_service.get_embedding(question)

        filters: Dict[str, Any] | None = {"book_id": book_id} if book_id is not None else None
        results = self.vector_repo.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            filters=filters,
        )

        return [
            {
                "id": r.id,
                "question": question,
                "score": r.score,
                "page_number": r.payload.get("page_number"),
                "text": r.payload.get("text"),
                
            }
            for r in results
        ]
        
    def delete_book_vectors(self, book_id: int):
        self.vector_repo.delete_by_filter(
            collection_name=self.collection_name,
            filters={"book_id": book_id},
        )

    def count_book_vectors(self, book_id: int) -> int:
        return self.vector_repo.count_by_filter(
            collection_name=self.collection_name,
            filters={"book_id": book_id},
        )

