from sentence_transformers import SentenceTransformer

class EmbeddingProvider:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> list:
        return self.model.encode(text).tolist()

    def get_embeddings(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return vectors.tolist()