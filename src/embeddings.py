from __future__ import annotations

from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper around a SentenceTransformer embedding model.

    This class keeps embedding logic in one place so you can swap
    models later without changing the rest of the project.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the embedding model.

        Good starter model:
        - all-MiniLM-L6-v2
          Fast, small, and good enough for coursework/demo RAG.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a list of texts into embeddings.

        Returns:
            A list of embedding vectors, one per input text.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        # Convert numpy arrays to plain Python lists for compatibility
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """
        Convert a single query into one embedding vector.
        """
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        return embedding.tolist()