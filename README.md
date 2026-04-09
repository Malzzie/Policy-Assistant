# Company Policy RAG Assistant

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that answers questions based on internal company policy documents.

The system:
- Loads policy documents (Markdown, HTML, PDF)
- Splits them into structured chunks
- Generates embeddings using Sentence Transformers
- Stores embeddings in a Chroma vector database
- Retrieves relevant chunks for a query
- Generates answers with citations (or fallback mode if LLM is unavailable)

---

## Features

- Multi-format document ingestion
- Semantic search using embeddings
- Context-aware question answering
- Citation-based responses
- Guardrails for safe and scoped answers
- Flask web application UI
- Evaluation pipeline with metrics
- CI/CD with GitHub Actions

---

## Tech Stack

- Python
- Flask
- ChromaDB
- Sentence Transformers
- OpenAI API (optional)
- PyTest

---

## Project Structure

rag-policy-assistant/
- app.py
- src/
- data/policies/
- eval/
- templates/
- static/
- tests/
- chroma_db/

---

## Setup Instructions

### 1. Clone the repository

    git clone <your-repo-link>
    cd rag-policy-assistant

### 2. Install dependencies

    pip install -r requirements.txt

---

## Environment Variables

Set your OpenAI API key (optional):

### Windows CMD
    set OPENAI_API_KEY=your_key_here

### PowerShell
    $env:OPENAI_API_KEY="your_key_here"

If not set, the system runs in **fallback mode**.

---

## How to Ingest Documents

    python -m src.ingest

This will:
- Load documents from data/policies/
- Chunk them
- Generate embeddings
- Store them in chroma_db/

---

## How to Run the App

    python app.py

Open in browser:
    http://127.0.0.1:5000/

---

## How to Run Tests

    pytest -q

---

## How to Run Evaluation

    python -m src.evaluation

Outputs:
- eval/eval_results_topk3.csv
- eval/eval_results_topk5.csv
- eval/latency_results_topk3.csv
- eval/latency_results_topk5.csv

---

## Example Questions

- How many PTO days can employees carry over?
- What are remote work core hours?
- How quickly must security incidents be reported?

---

## Notes

- The system enforces guardrails to restrict answers to policy-related questions
- If the OpenAI API is unavailable, fallback mode ensures system reliability

---

## Demo

GitHub Repo: <insert link>  
Demo Video: <insert link>
