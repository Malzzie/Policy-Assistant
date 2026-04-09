from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import time

from flask import Flask, jsonify, render_template, request

from src.rag_chain import PolicyRAGChain


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

rag = PolicyRAGChain(
    chroma_dir=CHROMA_DIR,
    collection_name="policy_chunks",
    embedding_model_name="all-MiniLM-L6-v2",
    llm_model="gpt-4o-mini",
)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "rag-policy-assistant",
        }
    )


@app.post("/chat")
def chat():
    try:
        data: Dict[str, Any] = request.get_json(silent=True) or {}
        question = str(data.get("question", "")).strip()

        if not question:
            return jsonify({"error": "Question is required."}), 400

        start_time = time.perf_counter()
        result = rag.answer_question(question=question, top_k=5)
        latency_seconds = time.perf_counter() - start_time

        return jsonify(
            {
                "question": question,
                "answer": result.get("answer", ""),
                "citations": result.get("citations", []),
                "mode": result.get("mode", "unknown"),
                "latency_seconds": round(latency_seconds, 3),
            }
        )

    except Exception as error:
        return jsonify(
            {
                "error": "An error occurred while processing the request.",
                "details": str(error),
            }
        ), 500


if __name__ == "__main__":
    app.run(debug=True)