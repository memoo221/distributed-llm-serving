
from rag.services.qdrant_service import VectorDBService


class PromptBuilder:
    def __init__(self, vector_service: VectorDBService):
        self.vector_service = vector_service

    def build_prompt(self, question: str, book_id: int | None = None, top_k: int = 3) -> str:
        search_results = self.vector_service.search_in_book(book_id, question, top_k)

        context = "\n\n".join(
            f"{r['text']}" for r in search_results
        )

        prompt = (
            "Here is the context you should use to answer the question:\n"
            f"{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return prompt