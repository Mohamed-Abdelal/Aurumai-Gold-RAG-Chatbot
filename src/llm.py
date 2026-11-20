"""Groq LLM client for AurumAI supporting RAG-grounded and Without-RAG baseline generation."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from groq import Groq, RateLimitError

from src.config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TEMPERATURE,
)
from src.retriever import RetrievedChunk


RAG_SYSTEM_PROMPT = """You are AurumAI, a specialized Gold Investment Recommendation Assistant.

STRICT GROUNDING & ADVISORY RULES:
1. Base your answer and investment recommendations STRICTLY on the retrieved gold-investing articles provided in the context below. Do NOT use external knowledge or invent numbers, fees, percentages, or products not present in the context.
2. Tailor recommendations directly to the user's stated budget, time horizon, risk profile, and region (e.g., Egypt vs. global) whenever provided.
3. Cite exact source filenames inline for every key claim or figure (for example: `[16_egypt_24k_gold_bars_vs_21k_gold_pounds.txt]` or `[20_gold_investor_decision_matrix.txt]`).
4. OUT-OF-SCOPE & UNANSWERABLE REFUSAL: If the user asks about a topic outside gold investing (such as cryptocurrencies, real estate mortgages, individual non-gold stocks, or personal tax filing) OR if the retrieved excerpts do not contain sufficient evidence to answer the question, explicitly refuse and state that the AurumAI knowledge base does not cover that topic. Never guess or fabricate advice.
5. Keep your response clear, actionable, structured, and concise.
6. Always end every response with this exact one-line disclaimer:
   *Disclaimer: This information is for educational purposes only and does not constitute licensed financial advice.*"""


NO_RAG_SYSTEM_PROMPT = """You are a general financial assistant. Answer the user's question concisely using only your pretrained knowledge."""


@dataclass
class LLMResponse:
    """Structured response returned by the Groq LLM."""

    answer: str
    model: str
    mode: str  # "rag" or "no_rag"
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


def format_retrieved_context(retrieved_chunks: List[RetrievedChunk]) -> str:
    """Format Top-K retrieved chunks into a clean, filename-labeled context block."""
    if not retrieved_chunks:
        return "No relevant articles were retrieved from the AurumAI knowledge base."

    blocks: List[str] = []
    for item in retrieved_chunks:
        c = item.chunk
        header = (
            f"[Source {item.rank}] Filename: {c.filename} | Title: {c.title} | "
            f"Category: {c.category} | Date: {c.date} | Source: {c.source} | "
            f"Cosine Similarity: {item.score:.4f}"
        )
        blocks.append(f"{header}\n{c.text}")

    return "\n\n---\n\n".join(blocks)


class GroqLLMClient:
    """Wrapper over the Groq API for fast free-tier LLM inference."""

    def __init__(self, api_key: Optional[str] = None):
        resolved_key = api_key or os.getenv("GROQ_API_KEY", "").strip()
        if not resolved_key:
            raise ValueError(
                "Missing GROQ_API_KEY. Set it in .env, Streamlit secrets, or the sidebar."
            )
        self.client = Groq(api_key=resolved_key)

    def _call_groq(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_completion_tokens: int,
        mode: str,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> LLMResponse:
        extra_args: Dict = {}
        if model.startswith("openai/gpt-oss"):
            extra_args["reasoning_effort"] = reasoning_effort

        start = time.perf_counter()
        last_err: Optional[Exception] = None
        completion = None
        budget = max_completion_tokens

        for attempt in range(4):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=budget,
                    **extra_args,
                )
            except RateLimitError as exc:
                last_err = exc
                time.sleep(5.0 * (attempt + 1))
                continue

            # Retry once with a larger token budget if hidden reasoning consumed the budget
            choice = completion.choices[0]
            if not (choice.message.content or "").strip() and choice.finish_reason == "length":
                budget = min(budget * 2, 2048)
                completion = None
                continue
            break

        if completion is None:
            raise last_err or RuntimeError("Groq API call failed after retries.")
        latency_ms = (time.perf_counter() - start) * 1000.0

        answer = completion.choices[0].message.content or ""
        # Strip inline <think> tags if emitted by Qwen reasoning models
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        return LLMResponse(
            answer=answer.strip(),
            model=model,
            mode=mode,
            latency_ms=round(latency_ms, 1),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def generate_with_rag(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
        model: str = DEFAULT_GROQ_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> LLMResponse:
        """Generate an answer grounded strictly in the Top-K retrieved gold-investing chunks."""
        context_block = format_retrieved_context(retrieved_chunks)
        messages: List[Dict[str, str]] = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]

        # Include recent conversational turns (last 4 messages) for follow-up coherence
        if chat_history:
            for msg in chat_history[-4:]:
                if msg.get("role") in ("user", "assistant") and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})

        user_prompt = (
            f"RETRIEVED ARTICLES FROM AURUMAI KNOWLEDGE BASE:\n"
            f"{context_block}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            f"Answer strictly from the retrieved articles, cite exact source filenames, "
            f"refuse if unanswerable or out of scope, and include the one-line educational disclaimer."
        )
        messages.append({"role": "user", "content": user_prompt})

        return self._call_groq(
            messages=messages,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            mode="rag",
        )

    def generate_without_rag(
        self,
        question: str,
        model: str = DEFAULT_GROQ_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> LLMResponse:
        """Generate a baseline answer WITHOUT any retrieved context."""
        messages: List[Dict[str, str]] = [{"role": "system", "content": NO_RAG_SYSTEM_PROMPT}]

        if chat_history:
            for msg in chat_history[-4:]:
                if msg.get("role") in ("user", "assistant") and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": question})

        return self._call_groq(
            messages=messages,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            mode="no_rag",
        )
