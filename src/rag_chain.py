from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import os

from src.retriever import PolicyRetriever
from src.prompts import build_system_prompt

from openai import OpenAI

from src.guardrails import (
    apply_output_guardrails,
    build_refusal_response,
    is_out_of_scope_question,
)


class PolicyRAGChain:
    """
    End-to-end RAG chain for policy question answering.

    Flow:
    1. Retrieve top-k chunks from Chroma
    2. Build a grounded prompt
    3. Try to send the prompt to the LLM
    4. If the LLM fails, fall back to retrieval-only answering
    5. Return a structured answer with citations
    """

    def __init__(
        self,
        chroma_dir: str | Path,
        collection_name: str = "policy_chunks",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self.retriever = PolicyRetriever(
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            model_name=embedding_model_name,
        )

        self.llm_model = llm_model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def build_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Convert retrieved chunks into one text block for the LLM.
        """
        context_parts: List[str] = []

        for index, item in enumerate(retrieved_chunks, start=1):
            metadata = item["metadata"]

            block = (
                f"[Chunk {index}]\n"
                f"Title: {metadata.get('title')}\n"
                f"Source: {metadata.get('source')}\n"
                f"Section: {metadata.get('section')}\n"
                f"Page: {metadata.get('page')}\n"
                f"Chunk ID: {metadata.get('chunk_id')}\n"
                f"Text: {item['text']}\n"
            )
            context_parts.append(block)

        return "\n" + ("\n" + ("-" * 80) + "\n").join(context_parts)

    def build_user_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Build the grounded user prompt.
        """
        return f"""Question:
{question}

Context:
{context}

Answer the question using only the context above.
Return valid JSON only.
"""

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Call the LLM and return the raw text response.
        """
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        response = self.client.chat.completions.create(
            model=self.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned an empty response.")

        return content.strip()

    def safe_json_parse(
        self,
        text: str,
        fallback_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Parse model output as JSON safely.
        """
        cleaned = text.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned.removeprefix("```json").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```").strip()

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        try:
            parsed = json.loads(cleaned)

            if "answer" not in parsed:
                parsed["answer"] = "The model returned JSON, but no answer field was found."

            if "citations" not in parsed or not isinstance(parsed["citations"], list):
                parsed["citations"] = []

            return parsed

        except json.JSONDecodeError:
            fallback_citations = self.build_citations(fallback_chunks, max_citations=2)

            return {
                "answer": (
                    "The model response could not be parsed as valid JSON. "
                    "A retrieval-based fallback answer should be used instead."
                ),
                "citations": fallback_citations,
                "raw_output": text,
            }

    def build_citations(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        max_citations: int = 2,
    ) -> List[Dict[str, str]]:
        """
        Build citations from retrieved chunks.
        """
        citations: List[Dict[str, str]] = []

        for item in retrieved_chunks[:max_citations]:
            metadata = item["metadata"]
            citations.append(
                {
                    "title": str(metadata.get("title", "")),
                    "source": str(metadata.get("source", "")),
                    "snippet": item["text"][:240].replace("\n", " ").strip(),
                }
            )

        return citations

    def build_fallback_answer(
       self,
       question: str,
       retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
       """
       Build a simple retrieval-only answer when the LLM is unavailable.
       """
       if not retrieved_chunks:
           return {
              "answer": (
                "I could not find any relevant policy content for this question "
                "in the indexed documents."
            ),
            "citations": [],
            "mode": "fallback",
            }

       citations = self.build_citations(retrieved_chunks, max_citations=2)
       top_snippet = citations[0]["snippet"] if citations else ""

       answer = (
        "The LLM is currently unavailable, so this answer is based directly on "
        f"the most relevant retrieved policy text: {top_snippet}"
      )

       return {
        "answer": answer,
        "citations": citations,
        "mode": "fallback",
      }

    def answer_question(
    self,
    question: str,
    top_k: int = 5,
) -> Dict[str, Any]:
        """
        End-to-end RAG call with smart fallback and guardrails.

        Returns:
        {
            "answer": "...",
            "citations": [...],
            "mode": "llm" or "fallback" or guardrail mode,
            "retrieved_chunks": [...]
        }
        """
        if not question or not question.strip():
           raise ValueError("Question cannot be empty.")

        # Guardrail 1: refuse out-of-scope questions early
        if is_out_of_scope_question(question):
           refusal = build_refusal_response()
           refusal["retrieved_chunks"] = []
           return refusal

        retrieved_chunks = self.retriever.retrieve(question=question, top_k=top_k)

        system_prompt = build_system_prompt()
        context = self.build_context(retrieved_chunks)
        user_prompt = self.build_user_prompt(question=question, context=context)

        try:
            raw_output = self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

            parsed_output = self.safe_json_parse(
            text=raw_output,
            fallback_chunks=retrieved_chunks,
        )

            # If parse failed in a recoverable way, switch to fallback answer
            if parsed_output.get("answer", "").startswith(
            "The model response could not be parsed as valid JSON"
            ):
                parsed_output = self.build_fallback_answer(
                question=question,
                retrieved_chunks=retrieved_chunks,
                )
            else:
                parsed_output["mode"] = "llm"

        except Exception as error:
            print("LLM call failed, switching to fallback mode.")
            print(type(error).__name__)
            print(str(error))

            parsed_output = self.build_fallback_answer(
                question=question,
                retrieved_chunks=retrieved_chunks,
            )
       # Guardrail 2: require citations and cap answer length
        parsed_output = apply_output_guardrails(
        parsed_output,
        max_answer_chars=500,
    )

        parsed_output["retrieved_chunks"] = retrieved_chunks
        return parsed_output


def pretty_print_response(result: Dict[str, Any]) -> None:
    """
    Print the final RAG answer in a readable way.
    """
    print("=" * 100)
    print("MODE")
    print("=" * 100)
    print(result.get("mode", "unknown"))
    print()

    print("=" * 100)
    print("ANSWER")
    print("=" * 100)
    print(result.get("answer", "No answer returned."))
    print()

    print("=" * 100)
    print("CITATIONS")
    print("=" * 100)
    citations = result.get("citations", [])

    if not citations:
        print("No citations returned.")
    else:
        for index, citation in enumerate(citations, start=1):
            print(f"Citation {index}")
            print("Title:", citation.get("title"))
            print("Source:", citation.get("source"))
            print("Snippet:", citation.get("snippet"))
            print("-" * 100)


def main() -> None:
    """
    Quick terminal test for one question.
    """
    base_dir = Path(__file__).resolve().parents[1]
    chroma_dir = base_dir / "chroma_db"

    rag = PolicyRAGChain(
        chroma_dir=chroma_dir,
        collection_name="policy_chunks",
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model="gpt-4o-mini",
    )

    question = "How many PTO days can employees carry over?"
    result = rag.answer_question(question=question, top_k=5)
    pretty_print_response(result)


if __name__ == "__main__":
    main()