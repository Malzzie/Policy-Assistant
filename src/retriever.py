from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.embeddings import EmbeddingModel
from src.vector_store import ChromaVectorStore


class PolicyRetriever:
    """
    Retrieves the most relevant policy chunks from the Chroma vector store.

    This class is responsible only for retrieval.
    It does not generate final answers yet.
    """

    def __init__(
        self,
        chroma_dir: str | Path,
        collection_name: str = "policy_chunks",
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """
        Initialize the retriever with:
        - a persistent Chroma collection
        - an embedding model for query embeddings
        """
        self.embedding_model = EmbeddingModel(model_name=model_name)
        self.vector_store = ChromaVectorStore(
            persist_directory=chroma_dir,
            collection_name=collection_name,
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant chunks for a user question.

        Returns a list of dictionaries containing:
        - chunk_id
        - text
        - metadata
        - distance
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        results = self.vector_store.similarity_search(
            query=question,
            embedding_model=self.embedding_model,
            top_k=top_k,
        )

        return results

    def retrieve_for_context(
        self,
        question: str,
        top_k: int = 5,
    ) -> str:
        """
        Retrieve top-k chunks and combine them into one context string.

        This is useful for the next step when you build the QA layer.
        """
        results = self.retrieve(question=question, top_k=top_k)

        context_parts: List[str] = []

        for index, result in enumerate(results, start=1):
            metadata = result["metadata"]

            block = (
                f"[Chunk {index}]\n"
                f"Title: {metadata.get('title')}\n"
                f"Source: {metadata.get('source')}\n"
                f"Section: {metadata.get('section')}\n"
                f"Page: {metadata.get('page')}\n"
                f"Chunk ID: {metadata.get('chunk_id')}\n"
                f"Text:\n{result['text']}\n"
            )
            context_parts.append(block)

        return "\n" + ("\n" + ("-" * 80) + "\n").join(context_parts)

    def pretty_print_results(
        self,
        question: str,
        top_k: int = 5,
    ) -> None:
        """
        Print retrieval results in a readable format for testing.
        """
        results = self.retrieve(question=question, top_k=top_k)

        print("=" * 100)
        print("QUESTION:", question)
        print(f"TOP {len(results)} RESULTS")
        print("=" * 100)

        for index, result in enumerate(results, start=1):
            metadata = result["metadata"]

            print(f"\nResult {index}")
            print("-" * 100)
            print("Distance:", result["distance"])
            print("Chunk ID:", result["chunk_id"])
            print("Doc ID:", metadata.get("doc_id"))
            print("Title:", metadata.get("title"))
            print("Source:", metadata.get("source"))
            print("Section:", metadata.get("section"))
            print("Page:", metadata.get("page"))
            print("Text Preview:")
            print(result["text"][:500].replace("\n", " "))
            print("-" * 100)


def main() -> None:
    """
    Small direct test when running this file on its own.
    """
    base_dir = Path(__file__).resolve().parents[1]
    chroma_dir = base_dir / "chroma_db"

    retriever = PolicyRetriever(chroma_dir=chroma_dir)

    sample_question = "How many PTO days can employees carry over?"
    retriever.pretty_print_results(sample_question, top_k=5)


if __name__ == "__main__":
    main()