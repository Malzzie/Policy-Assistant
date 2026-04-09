from __future__ import annotations

from pathlib import Path

from src.rag_chain import PolicyRAGChain, pretty_print_response


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    chroma_dir = base_dir / "chroma_db"

    rag = PolicyRAGChain(
        chroma_dir=chroma_dir,
        collection_name="policy_chunks",
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model="gpt-4o-mini",
    )

    test_questions = [
        # In-scope
        "How many PTO days can employees carry over?",
        "What are the remote work core hours?",
        "How quickly must security incidents be reported?",
        "What should employees do if a company device is lost or stolen?",
        # Out-of-scope
        "Who won the FIFA World Cup in 2010?",
        "What is the weather in Cape Town today?",
        "Write me a poem about success.",
        "How do I invest R10,000 in South Africa?",
    ]

    for question in test_questions:
        print("\n" + "#" * 100)
        print("QUESTION:", question)
        print("#" * 100 + "\n")

        result = rag.answer_question(question=question, top_k=5)
        pretty_print_response(result)


if __name__ == "__main__":
    main()