"""Local vector store for agency knowledge."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .chunking import chunk_documents, load_documents
from .models import KnowledgeChunk, RetrievedChunk


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


class KnowledgeBase:
    """A small local vector store backed by pure-Python TF-IDF vectors.

    This keeps the system runnable in a fresh environment without external
    embedding APIs. Agencies can later swap this class for a hosted vector DB
    or neural embeddings while preserving the generator interface.
    """

    def __init__(self, chunks: Iterable[KnowledgeChunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("KnowledgeBase requires at least one knowledge chunk")

        tokenized_chunks = [self._terms(chunk.text) for chunk in self.chunks]
        document_frequency = Counter(
            term for terms in tokenized_chunks for term in set(terms)
        )
        document_count = len(tokenized_chunks)
        self.idf = {
            term: math.log((1 + document_count) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }
        self.vectors = [self._tfidf_vector(terms) for terms in tokenized_chunks]

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[str | Path],
        chunk_size: int = 900,
        overlap: int = 120,
    ) -> "KnowledgeBase":
        """Create a knowledge base from Markdown, text, and CSV files."""

        normalized = [Path(path) for path in paths]
        documents = load_documents(normalized)
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
        return cls(chunks)

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Retrieve chunks that best match a prospect query."""

        if not query.strip():
            return []

        query_vector = self._tfidf_vector(self._terms(query))
        if not query_vector:
            return []

        scores = [
            self._cosine_similarity(query_vector, vector)
            for vector in self.vectors
        ]
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )[:top_k]

        results: list[RetrievedChunk] = []
        for idx in ranked_indices:
            score = scores[idx]
            if score <= 0:
                continue
            chunk = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    id=chunk.id,
                    source=chunk.source,
                    text=chunk.text,
                    score=score,
                )
            )

        return results

    def _terms(self, text: str) -> list[str]:
        tokens = [
            token
            for token in TOKEN_RE.findall(text.lower())
            if token not in STOP_WORDS and len(token) > 1
        ]
        bigrams = [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]
        return tokens + bigrams

    def _tfidf_vector(self, terms: list[str]) -> dict[str, float]:
        counts = Counter(term for term in terms if term in self.idf)
        if not counts:
            return {}

        total = sum(counts.values())
        weights = {
            term: (count / total) * self.idf[term]
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(weight * weight for weight in weights.values()))
        if norm == 0:
            return {}
        return {term: weight / norm for term, weight in weights.items()}

    @staticmethod
    def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(weight * right.get(term, 0.0) for term, weight in left.items())
