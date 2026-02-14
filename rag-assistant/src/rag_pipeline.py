"""
RAG Pipeline Module
Orchestrates the retrieval-augmented generation workflow.
Supports: OpenAI, Anthropic, and local/Ollama models.
"""

import os
import json
import time
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime

from src.vector_store import SearchResult


# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────

@dataclass
class RAGResponse:
    """Represents a complete RAG response."""
    answer: str
    sources: List[SearchResult]
    query: str
    model: str
    context_used: str
    latency_ms: float
    timestamp: str = ""
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def format_sources(self) -> str:
        """Format sources for display."""
        if not self.sources:
            return "No sources found."

        lines = []
        for i, src in enumerate(self.sources, 1):
            source_name = src.metadata.get("source", "Unknown")
            chunk_idx = src.metadata.get("chunk_index", "?")
            lines.append(
                f"  [{i}] {src.relevance_label} (score: {src.score:.3f})\n"
                f"      Source: {source_name} | Chunk: {chunk_idx}\n"
                f"      Preview: {src.content[:150].strip()}..."
            )
        return "\n".join(lines)


# ──────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────

class PromptTemplates:
    """Collection of prompt templates for different RAG tasks."""

    ANSWER_QUESTION = """You are a helpful document intelligence assistant. Answer the user's question based ONLY on the provided context. If the context doesn't contain enough information to fully answer the question, say so clearly.

## Rules:
1. Only use information from the provided context
2. If the context is insufficient, clearly state what information is missing
3. Cite which parts of the context support your answer
4. Be precise, clear, and well-structured
5. If multiple sources provide different information, note the discrepancies

## Context:
{context}

## Question:
{question}

## Answer:"""

    SUMMARIZE = """You are a document intelligence assistant. Provide a comprehensive summary of the following document context.

## Rules:
1. Cover all main topics and key points
2. Organize the summary logically
3. Highlight important facts, figures, or conclusions
4. Keep the summary concise but thorough

## Context:
{context}

## Summary:"""

    COMPARE = """You are a document intelligence assistant. Compare and analyze the information from different document sections provided in the context.

## Context:
{context}

## Comparison Request:
{question}

## Analysis:"""

    EXTRACT = """You are a document intelligence assistant. Extract specific information as requested from the provided context.

## Context:
{context}

## Extraction Request:
{question}

## Extracted Information (as structured JSON):"""


# ──────────────────────────────────────────────
# LLM Providers
# ──────────────────────────────────────────────

class LLMProvider:
    """Base class for LLM providers."""

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI GPT models."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return response.choices[0].message.content

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicProvider(LLMProvider):
    """Anthropic Claude models."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        from anthropic import Anthropic

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text


class OllamaProvider(LLMProvider):
    """Ollama local models (Llama, Mistral, etc.)."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                },
            },
        )
        response.raise_for_status()
        return response.json()["response"]

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        import requests

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                },
            },
            stream=True,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]


# ──────────────────────────────────────────────
# RAG Pipeline
# ──────────────────────────────────────────────

class RAGPipeline:
    """
    Main RAG orchestration pipeline.
    Retrieves relevant context → Builds prompt → Generates answer.
    """

    def __init__(
        self,
        vector_store,
        llm_provider: LLMProvider,
        top_k: int = 5,
        score_threshold: float = 0.1,
        max_context_length: int = 4000,
    ):
        self.vector_store = vector_store
        self.llm = llm_provider
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.max_context_length = max_context_length
        self.conversation_history: List[Dict] = []

    def _build_context(self, results: List[SearchResult]) -> str:
        """Build context string from search results, respecting length limits."""
        context_parts = []
        total_length = 0

        for i, result in enumerate(results, 1):
            source = result.metadata.get("source", "Unknown")
            chunk_idx = result.metadata.get("chunk_index", "?")
            header = f"[Source {i}: {source}, Chunk {chunk_idx}, Relevance: {result.score:.2f}]"
            section = f"{header}\n{result.content}"

            if total_length + len(section) > self.max_context_length:
                remaining = self.max_context_length - total_length
                if remaining > 100:
                    section = section[:remaining] + "... [truncated]"
                    context_parts.append(section)
                break

            context_parts.append(section)
            total_length += len(section)

        return "\n\n---\n\n".join(context_parts)

    def query(
        self,
        question: str,
        mode: str = "answer",  # "answer", "summarize", "compare", "extract"
        filter_doc_id: Optional[str] = None,
        top_k: Optional[int] = None,
        stream: bool = False,
    ) -> RAGResponse:
        """
        Execute a RAG query.

        Args:
            question: The user's question
            mode: Type of response ("answer", "summarize", "compare", "extract")
            filter_doc_id: Optional document ID to restrict search to
            top_k: Override default top_k
            stream: Whether to stream the response (not used in this method)

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        start_time = time.time()

        # Step 1: Retrieve relevant chunks
        k = top_k or self.top_k
        results = self.vector_store.search(
            query=question,
            top_k=k,
            score_threshold=self.score_threshold,
            filter_doc_id=filter_doc_id,
        )

        if not results:
            return RAGResponse(
                answer="I couldn't find any relevant information in the uploaded documents to answer your question. Please try rephrasing or upload relevant documents.",
                sources=[],
                query=question,
                model=getattr(self.llm, 'model', 'unknown'),
                context_used="",
                latency_ms=0,
            )

        # Step 2: Build context
        context = self._build_context(results)

        # Step 3: Select and fill prompt template
        templates = {
            "answer": PromptTemplates.ANSWER_QUESTION,
            "summarize": PromptTemplates.SUMMARIZE,
            "compare": PromptTemplates.COMPARE,
            "extract": PromptTemplates.EXTRACT,
        }
        template = templates.get(mode, PromptTemplates.ANSWER_QUESTION)
        prompt = template.format(context=context, question=question)

        # Step 4: Generate answer
        answer = self.llm.generate(prompt)

        latency = (time.time() - start_time) * 1000

        # Step 5: Build response
        response = RAGResponse(
            answer=answer,
            sources=results,
            query=question,
            model=getattr(self.llm, 'model', 'unknown'),
            context_used=context,
            latency_ms=latency,
            metadata={
                "mode": mode,
                "top_k": k,
                "num_sources": len(results),
                "avg_relevance": sum(r.score for r in results) / len(results) if results else 0,
            },
        )

        # Track conversation
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "sources_count": len(results),
            "timestamp": response.timestamp,
        })

        return response

    def query_stream(
        self,
        question: str,
        mode: str = "answer",
        filter_doc_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Generator:
        """Stream a RAG query response."""
        k = top_k or self.top_k
        results = self.vector_store.search(
            query=question,
            top_k=k,
            score_threshold=self.score_threshold,
            filter_doc_id=filter_doc_id,
        )

        if not results:
            yield "I couldn't find any relevant information in the uploaded documents."
            return

        context = self._build_context(results)
        templates = {
            "answer": PromptTemplates.ANSWER_QUESTION,
            "summarize": PromptTemplates.SUMMARIZE,
            "compare": PromptTemplates.COMPARE,
            "extract": PromptTemplates.EXTRACT,
        }
        template = templates.get(mode, PromptTemplates.ANSWER_QUESTION)
        prompt = template.format(context=context, question=question)

        yield from self.llm.generate_stream(prompt)

    def get_conversation_history(self) -> List[Dict]:
        """Return the conversation history."""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
