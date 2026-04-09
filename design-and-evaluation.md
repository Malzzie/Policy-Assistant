
---

# 🧠 2. design-and-evaluation.md (FULL RUBRIC MATCH)

```markdown
# Design and Evaluation

## System Architecture

The system follows a Retrieval-Augmented Generation (RAG) architecture:

1. Document ingestion
2. Chunking with metadata
3. Embedding using Sentence Transformers
4. Storage in ChromaDB
5. Retrieval of top-k relevant chunks
6. Answer generation (LLM or fallback)
7. Guardrails enforcement

---

## Why Flask

Flask was chosen because:
- Lightweight and easy to integrate
- Minimal setup for APIs and UI
- Suitable for small demo applications
- Provides flexibility for custom endpoints

---

## Why ChromaDB

ChromaDB was selected because:
- Simple to set up and use locally
- Supports persistent vector storage
- Integrates easily with Python
- Good performance for small to medium datasets

---

## Why Sentence Transformers

The model `all-MiniLM-L6-v2` was used because:
- Fast and lightweight
- Good semantic similarity performance
- Suitable for CPU-based environments
- Easy integration with Python

---

## Chunking Strategy

- Chunk size: 700 characters
- Overlap: 100 characters
- Markdown uses heading-aware splitting
- Other formats use fallback character-based splitting

Reason:
- Ensures context continuity
- Balances retrieval precision and coverage

---

## Retrieval Strategy

- Vector similarity search
- top-k = 5 (baseline)
- compared with top-k = 3

Reason:
- top-k=5 improves recall
- top-k=3 improves precision

---

## Prompting Strategy

- System prompt enforces:
  - answer only from context
  - no hallucination
  - structured JSON output
- User prompt includes:
  - question
  - retrieved context

---

## Guardrails

The system includes:

- Out-of-scope detection (keyword-based)
- Refusal of non-policy questions
- Mandatory citations for answers
- Answer length limit
- Fallback mode when LLM is unavailable

---

## Evaluation Method

Dataset:
- 31 questions across multiple policy topics

Metrics:

### Groundedness %
- Checks if answer includes citations

### Citation Accuracy %
- Measures overlap between citation snippet and gold answer

### Latency
- p50 (median)
- p95 (high percentile)

---

## Results

| Metric | Value |
|------|------|
| Groundedness | 93.55% |
| Citation Accuracy | 77.42% |
| Latency p50 | 2.4182 seconds |
| Latency p95 | 2.8650 seconds |

---

## Observations

- Increasing top-k improves recall but may reduce precision
- Chunking strategy significantly impacts retrieval quality
- Fallback mode ensures system reliability without API dependency

---

## Limitations

- Guardrails use simple keyword matching
- Evaluation uses token overlap instead of semantic similarity
- Performance depends on document structure and embedding quality

---

## Improvements

- Use semantic evaluation metrics (e.g., cosine similarity scoring)
- Improve guardrails with LLM-based classification
- Add reranking for better retrieval accuracy
- Improve UI with conversation history and filtering