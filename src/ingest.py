from __future__ import annotations

from pathlib import Path

from src.loaders import load_documents
from src.chunking import chunk_documents
from src.embeddings import EmbeddingModel
from src.vector_store import ChromaVectorStore


def main() -> None:
    """
    End-to-end ingestion script.

    Steps:
    1. Load policy files
    2. Chunk the documents
    3. Create embeddings
    4. Store them in Chroma
    5. Run a small test search
    """
    base_dir = Path(__file__).resolve().parents[1]
    policies_dir = base_dir / "data" / "policies"
    chroma_dir = base_dir / "chroma_db"

    print("Base directory:", base_dir)
    print("Policies directory:", policies_dir)
    print("Chroma directory:", chroma_dir)
    print("-" * 80)

    # Step 1: Load documents
    docs = load_documents(policies_dir)
    print(f"Loaded {len(docs)} source documents/pages")

    # Step 2: Chunk documents
    chunks = chunk_documents(
        docs=docs,
        chunk_size=700,
        chunk_overlap=100,
    )
    print(f"Created {len(chunks)} chunks")

    if not chunks:
        print("No chunks were created. Stopping.")
        return

    # Step 3: Create embedding model
    embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")

    # Step 4: Create vector store
    vector_store = ChromaVectorStore(
        persist_directory=chroma_dir,
        collection_name="policy_chunks",
    )

    # During development, it is often easier to rebuild cleanly
    vector_store.reset_collection()

    # Add chunks into Chroma
    vector_store.add_chunks(
        chunks=chunks,
        embedding_model=embedding_model,
        batch_size=32,
    )

    print(f"Stored chunk count in Chroma: {vector_store.count()}")
    print("-" * 80)

    # Step 5: Verify search
    test_query = "How many PTO days can employees carry over?"
    print("Test query:", test_query)

    results = vector_store.similarity_search(
        query=test_query,
        embedding_model=embedding_model,
        top_k=5,
    )

    print(f"Top {len(results)} results:\n")

    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(f"Result {index}")
        print("chunk_id:", result["chunk_id"])
        print("distance:", result["distance"])
        print("doc_id:", metadata.get("doc_id"))
        print("title:", metadata.get("title"))
        print("source:", metadata.get("source"))
        print("section:", metadata.get("section"))
        print("page:", metadata.get("page"))
        print("text preview:", result["text"][:250].replace("\n", " "))
        print("-" * 80)


if __name__ == "__main__":
    main()