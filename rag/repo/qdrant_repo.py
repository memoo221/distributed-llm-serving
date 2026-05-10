from typing import List, Dict, Any
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    VectorParams,
    Distance,
)
from rag.Clients.qdrant_client import QdrantClientManager



class QdrantVectorRepository():

    def __init__(self):
        self.client = QdrantClientManager().client

    def ensure_collection_exists(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
    ) -> None:
        collections = self.client.get_collections().collections
        existing_names = [c.name for c in collections]

        if collection_name in existing_names:
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance,
            ),
        )


    def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:

        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i],
            )
            for i in range(len(vectors))
        ]

        self.client.upsert(
            collection_name=collection_name,
            points=points,
        )
        
        
        
        
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        filters: Dict[str, Any] | None = None,
    ):

        qdrant_filter = None

        if filters:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key=k,
                        match=MatchValue(value=v),
                    )
                    for k, v in filters.items()
                ]
            )

        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
        )

        return response.points
    

    def delete_by_filter(self, collection_name: str, filters: Dict):

        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
                for key, value in filters.items()
            ]
        )

        self.client.delete(
            collection_name=collection_name,
            points_selector=qdrant_filter,
        )

    def count_by_filter(self, collection_name: str, filters: Dict) -> int:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
                for key, value in filters.items()
            ]
        )

        response = self.client.count(
            collection_name=collection_name,
            count_filter=qdrant_filter,
            exact=True,
        )
        return response.count
