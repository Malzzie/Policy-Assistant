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

    questions = [
        "How many PTO days can employees carry over?",
        "Who is eligible for remote work?",
        "What are the core working hours for remote employees?",
        "How soon must expenses be submitted?",
        "What should employees do if a company device is lost or stolen?",
        "How quickly must security incidents be reported?",
        "What behavior is prohibited under the code of conduct?",
        "Can employees store confidential data on personal devices?",
        "Are employees allowed limited personal use of company devices?",
        "Does the company allow additional unpaid leave for religious observances?",
    ]

    for question in questions:
        print("\n" + "#" * 100)
        print("QUESTION:", question)
        print("#" * 100 + "\n")

        result = rag.answer_question(question=question, top_k=5)
        pretty_print_response(result)


if __name__ == "__main__":
    main()