"""End-to-end RAG pipeline combining Embeddings, Pure-NumPy Retrieval, and Groq LLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.config import (
    CHUNKS_PATH,
    DEFAULT_GROQ_MODEL,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    EMBEDDINGS_PATH,
)
from src.embeddings import EmbeddingModel
from src.llm import GroqLLMClient, LLMResponse
from src.retriever import NumPyVectorIndex, RetrievedChunk


@dataclass
class RAGResult:
    """Output of a single query through the RAG or Without-RAG pipeline."""

    question: str
    llm_response: LLMResponse
    retrieved_chunks: List[RetrievedChunk]


class RAGPipeline:
    """Coordinates query embedding, pure-NumPy Top-K retrieval, and Groq generation."""

    def __init__(
        self,
        index: NumPyVectorIndex,
        embedding_model: EmbeddingModel,
        llm_client: Optional[GroqLLMClient] = None,
    ):
        self.index = index
        self.embedding_model = embedding_model
        self.llm_client = llm_client

    @classmethod
    def from_disk(
        cls,
        embeddings_path: Path = EMBEDDINGS_PATH,
        chunks_path: Path = CHUNKS_PATH,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        groq_api_key: Optional[str] = None,
    ) -> "RAGPipeline":
        index = NumPyVectorIndex.load(embeddings_path, chunks_path)
        embedder = EmbeddingModel(model_name=embedding_model_name)
        llm = GroqLLMClient(api_key=groq_api_key) if groq_api_key else None
        return cls(index=index, embedding_model=embedder, llm_client=llm)

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> List[RetrievedChunk]:
        """Embed the query and fetch Top-K chunks via pure-NumPy cosine similarity."""
        query_vec = self.embedding_model.embed_query(question)
        return self.index.search(
            query_embedding=query_vec,
            top_k=top_k,
            min_similarity=min_similarity,
        )

    def _require_llm(self, llm_client: Optional[GroqLLMClient]) -> GroqLLMClient:
        client = llm_client or self.llm_client
        if client is None:
            raise ValueError("GroqLLMClient is not configured. Provide a GROQ_API_KEY.")
        return client

    def answer_with_rag(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        model: str = DEFAULT_GROQ_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        chat_history: Optional[List[Dict[str, str]]] = None,
        llm_client: Optional[GroqLLMClient] = None,
    ) -> RAGResult:
        """Execute full Retrieval-Augmented Generation."""
        client = self._require_llm(llm_client)
        retrieved = self.retrieve(
            question=question,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        response = client.generate_with_rag(
            question=question,
            retrieved_chunks=retrieved,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            chat_history=chat_history,
        )
        return RAGResult(question=question, llm_response=response, retrieved_chunks=retrieved)

    def answer_without_rag(
        self,
        question: str,
        model: str = DEFAULT_GROQ_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        chat_history: Optional[List[Dict[str, str]]] = None,
        llm_client: Optional[GroqLLMClient] = None,
    ) -> RAGResult:
        """Execute direct LLM generation without retrieval (baseline comparison)."""
        client = self._require_llm(llm_client)
        response = client.generate_without_rag(
            question=question,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            chat_history=chat_history,
        )
        return RAGResult(question=question, llm_response=response, retrieved_chunks=[])

    def compare(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        model: str = DEFAULT_GROQ_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        llm_client: Optional[GroqLLMClient] = None,
    ) -> Dict[str, RAGResult]:
        """Run both With-RAG and Without-RAG on the same question."""
        with_rag = self.answer_with_rag(
            question=question,
            top_k=top_k,
            min_similarity=min_similarity,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            llm_client=llm_client,
        )
        without_rag = self.answer_without_rag(
            question=question,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            llm_client=llm_client,
        )
        return {"with_rag": with_rag, "without_rag": without_rag}

    def get_stats(self) -> Dict:
        docs = self.index.get_unique_documents()
        return {
            "num_documents": len(docs),
            "num_chunks": self.index.num_chunks,
            "embedding_dim": self.index.embedding_dim,
            "embedding_model": self.embedding_model.model_name,
        }
