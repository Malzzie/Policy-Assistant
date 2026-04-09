from __future__ import annotations

from pathlib import Path

from src.retriever import PolicyRetriever


def main() -> None:
    """
    Manual retrieval test with 10 policy questions.
    """
    base_dir = Path(__file__).resolve().parents[1]
    chroma_dir = base_dir / "chroma_db"

    retriever = PolicyRetriever(chroma_dir=chroma_dir)

    questions = [
        "How many PTO days can employees carry over?",
        "Who is eligible for remote work?",
        "What are the core working hours for remote employees?",
        "What expenses can employees claim back?",
        "How soon must expenses be submitted?",
        "Which public holidays does the company recognize?",
        "What should an employee do if a company device is lost or stolen?",
        "What is the device encryption or password protection rule?",
        "How quickly must security incidents be reported?",
        "What behavior is prohibited under the code of conduct?",
    ]

    for question in questions:
        retriever.pretty_print_results(question=question, top_k=5)
        print("\n\n")


if __name__ == "__main__":
    main()