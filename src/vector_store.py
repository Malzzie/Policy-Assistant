from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection

from src.embeddings import EmbeddingModel


class ChromaVectorStore:
    """
    Thin wrapper around ChromaDB.

    Responsibilities:
    - create/open a persistent Chroma collection
    - add chunked documents
    - run similarity search
    """

    def __init__(
        self,
        persist_directory: str | Path = "chroma_db",
        collection_name: str = "policy_chunks",
    ) -> None:
        """
        Initialize persistent Chroma client and collection.
        """
        self.persist_directory = str(Path(persist_directory))
        self.collection_name = collection_name

        # Persistent client means the DB is saved to disk
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # We will provide embeddings ourselves, so no Chroma embedding function needed
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def reset_collection(self) -> None:
        """
        Delete and recreate the collection.

        Useful during development if you want to rebuild from scratch.
        """
        existing_collections = self.client.list_collections()
        existing_names = [collection.name for collection in existing_collections]

        if self.collection_name in existing_names:
            self.client.delete_collection(self.collection_name)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embedding_model: EmbeddingModel,
        batch_size: int = 32,
    ) -> None:
        """
        Add chunked documents to Chroma.

        Each chunk should have:
        - chunk_id
        - text
        - doc_id
        - title
        - source
        - section
        - page
        """
        if not chunks:
            print("No chunks to add.")
            return

        # Process in small batches so memory use stays manageable
        for start_index in range(0, len(chunks), batch_size):
            batch = chunks[start_index : start_index + batch_size]

            ids = [chunk["chunk_id"] for chunk in batch]
            documents = [chunk["text"] for chunk in batch]

            # Metadata stored alongside the vectors
            metadatas = []
            for chunk in batch:
                metadata = {
                    "doc_id": chunk.get("doc_id"),
                    "title": chunk.get("title"),
                    "source": chunk.get("source"),
                    "chunk_id": chunk.get("chunk_id"),
                    "section": chunk.get("section") if chunk.get("section") is not None else "",
                    "page": chunk.get("page") if chunk.get("page") is not None else -1,
                }
                metadatas.append(metadata)

            embeddings = embedding_model.embed_texts(documents)

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            print(
                f"Added batch {start_index + 1} to "
                f"{min(start_index + batch_size, len(chunks))}"
            )

    def similarity_search(
        self,
        query: str,
        embedding_model: EmbeddingModel,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search the vector store for chunks similar to the query.
        """
        query_embedding = embedding_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        output: List[Dict[str, Any]] = []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            output.append(
                {
                    "chunk_id": chunk_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return output

    def count(self) -> int:
        """
        Return number of items currently stored in the collection.
        """
        return self.collection.count()