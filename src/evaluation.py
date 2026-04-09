from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import csv
import math
import time

from src.rag_chain import PolicyRAGChain


def normalize_text(text: str) -> str:
    """
    Normalize text for lightweight string comparisons.
    """
    return " ".join(text.lower().strip().split())


def token_overlap_score(text_a: str, text_b: str) -> float:
    """
    Compute a simple token overlap score between two texts.

    Returns a value between 0.0 and 1.0.
    """
    tokens_a = set(normalize_text(text_a).split())
    tokens_b = set(normalize_text(text_b).split())

    if not tokens_a or not tokens_b:
        return 0.0

    overlap = tokens_a.intersection(tokens_b)
    return len(overlap) / max(1, len(tokens_b))


def has_usable_citations(result: Dict[str, Any]) -> bool:
    """
    Check if the result includes at least one non-empty citation.
    """
    citations = result.get("citations", [])
    if not isinstance(citations, list) or len(citations) == 0:
        return False

    for citation in citations:
        if citation.get("title") or citation.get("source") or citation.get("snippet"):
            return True

    return False


def is_grounded(result: Dict[str, Any]) -> bool:
    """
    Lightweight groundedness rule:
    - answer exists
    - at least one citation exists
    """
    answer = str(result.get("answer", "")).strip()
    return bool(answer) and has_usable_citations(result)


def citation_matches_gold(
    citations: List[Dict[str, Any]],
    gold_answer: str,
    threshold: float = 0.25,
) -> bool:
    """
    Check whether at least one citation snippet overlaps enough with the gold answer.
    """
    gold_answer = normalize_text(gold_answer)

    if not gold_answer:
        return False

    for citation in citations:
        snippet = normalize_text(str(citation.get("snippet", "")))
        if not snippet:
            continue

        score = token_overlap_score(snippet, gold_answer)
        if score >= threshold:
            return True

    return False


def percentile(values: List[float], p: float) -> float:
    """
    Compute a simple percentile without external dependencies.

    Example:
    - p=50 for p50
    - p=95 for p95
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_values[int(k)]

    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def load_eval_questions(csv_path: str | Path) -> List[Dict[str, str]]:
    """
    Load evaluation questions from CSV.

    Expected columns:
    - question
    - topic
    - gold_answer
    """
    csv_path = Path(csv_path)

    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "question": str(row.get("question", "")).strip(),
                    "topic": str(row.get("topic", "")).strip(),
                    "gold_answer": str(row.get("gold_answer", "")).strip(),
                }
            )

    return rows


def save_csv(rows: List[Dict[str, Any]], output_path: str | Path) -> None:
    """
    Save a list of dictionaries to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_eval_row(
    item: Dict[str, str],
    result: Dict[str, Any],
    latency_seconds: float,
) -> Dict[str, Any]:
    """
    Build one row for eval_results.csv
    """
    citations = result.get("citations", [])
    mode = result.get("mode", "unknown")
    answer = str(result.get("answer", "")).strip()

    grounded = is_grounded(result)
    citation_ok = citation_matches_gold(citations, item["gold_answer"])

    first_citation_title = ""
    first_citation_source = ""
    first_citation_snippet = ""

    if citations:
        first = citations[0]
        first_citation_title = str(first.get("title", ""))
        first_citation_source = str(first.get("source", ""))
        first_citation_snippet = str(first.get("snippet", ""))

    return {
        "question": item["question"],
        "topic": item["topic"],
        "gold_answer": item["gold_answer"],
        "predicted_answer": answer,
        "mode": mode,
        "citation_count": len(citations),
        "first_citation_title": first_citation_title,
        "first_citation_source": first_citation_source,
        "first_citation_snippet": first_citation_snippet,
        "grounded": grounded,
        "citation_match_gold": citation_ok,
        "latency_seconds": round(latency_seconds, 4),
    }


def build_latency_row(
    item: Dict[str, str],
    result: Dict[str, Any],
    latency_seconds: float,
) -> Dict[str, Any]:
    """
    Build one row for latency_results.csv
    """
    return {
        "question": item["question"],
        "topic": item["topic"],
        "mode": result.get("mode", "unknown"),
        "latency_seconds": round(latency_seconds, 4),
    }


def print_summary(
    eval_rows: List[Dict[str, Any]],
    latency_rows: List[Dict[str, Any]],
    label: str = "Evaluation",
) -> None:
    """
    Print summary metrics for your report/demo.
    """
    total = len(eval_rows)

    grounded_count = sum(1 for row in eval_rows if row["grounded"])
    citation_match_count = sum(1 for row in eval_rows if row["citation_match_gold"])

    latencies = [float(row["latency_seconds"]) for row in latency_rows]

    groundedness_pct = (grounded_count / total * 100) if total else 0.0
    citation_accuracy_pct = (citation_match_count / total * 100) if total else 0.0
    latency_p50 = percentile(latencies, 50)
    latency_p95 = percentile(latencies, 95)

    print("\n" + "=" * 80)
    print(f"{label.upper()} SUMMARY")
    print("=" * 80)
    print(f"Total questions: {total}")
    print(f"Groundedness %: {groundedness_pct:.2f}")
    print(f"Citation accuracy %: {citation_accuracy_pct:.2f}")
    print(f"Latency p50 (s): {latency_p50:.4f}")
    print(f"Latency p95 (s): {latency_p95:.4f}")
    print("=" * 80)


def run_evaluation(
    eval_csv_path: str | Path,
    eval_results_path: str | Path,
    latency_results_path: str | Path,
    top_k: int = 5,
) -> None:
    """
    Run the full evaluation pipeline.
    """
    base_dir = Path(__file__).resolve().parents[1]
    chroma_dir = base_dir / "chroma_db"

    rag = PolicyRAGChain(
        chroma_dir=chroma_dir,
        collection_name="policy_chunks",
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model="gpt-4o-mini",
    )

    eval_items = load_eval_questions(eval_csv_path)

    eval_rows: List[Dict[str, Any]] = []
    latency_rows: List[Dict[str, Any]] = []

    print(f"Loaded {len(eval_items)} evaluation questions.")
    print(f"Running evaluation with top_k={top_k}\n")

    for index, item in enumerate(eval_items, start=1):
        question = item["question"]

        print(f"[{index}/{len(eval_items)}] Evaluating: {question}")

        start = time.perf_counter()
        result = rag.answer_question(question=question, top_k=top_k)
        latency_seconds = time.perf_counter() - start

        eval_rows.append(
            build_eval_row(
                item=item,
                result=result,
                latency_seconds=latency_seconds,
            )
        )

        latency_rows.append(
            build_latency_row(
                item=item,
                result=result,
                latency_seconds=latency_seconds,
            )
        )

    save_csv(eval_rows, eval_results_path)
    save_csv(latency_rows, latency_results_path)

    print_summary(
        eval_rows=eval_rows,
        latency_rows=latency_rows,
        label=f"top_k={top_k}",
    )

    print(f"\nSaved evaluation results to: {eval_results_path}")
    print(f"Saved latency results to: {latency_results_path}")


def run_top_k_comparison() -> None:
    """
    Run evaluation twice:
    - once with top_k=3
    - once with top_k=5

    This lets you compare retrieval depth without rebuilding the vector store.
    """
    base_dir = Path(__file__).resolve().parents[1]
    eval_dir = base_dir / "eval"

    print("\nRunning top-k=3 evaluation")
    run_evaluation(
        eval_csv_path=eval_dir / "eval_questions.csv",
        eval_results_path=eval_dir / "eval_results_topk3.csv",
        latency_results_path=eval_dir / "latency_results_topk3.csv",
        top_k=3,
    )

    print("\nRunning top-k=5 evaluation")
    run_evaluation(
        eval_csv_path=eval_dir / "eval_questions.csv",
        eval_results_path=eval_dir / "eval_results_topk5.csv",
        latency_results_path=eval_dir / "latency_results_topk5.csv",
        top_k=5,
    )


def main() -> None:
    """
    Default Day 13 run.

    This runs the comparison between top-k=3 and top-k=5.
    """
    run_top_k_comparison()


if __name__ == "__main__":
    main()