"""Local vector store for agency knowledge."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import chunk_documents, load_documents
from .models import KnowledgeChunk, RetrievedChunk


class KnowledgeBase:
    """A small local vector store backed by scikit-learn TF-IDF vectors.

    This keeps the system runnable in a fresh environment without external
    embedding APIs. Agencies can later swap this class for a hosted vector DB
    or neural embeddings while preserving the generator interface.
    """

    def __init__(self, chunks: Iterable[KnowledgeChunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("KnowledgeBase requires at least one knowledge chunk")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=30_000,
        )
        self.matrix = self.vectorizer.fit_transform(chunk.text for chunk in self.chunks)

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

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indices = scores.argsort()[::-1][:top_k]

        results: list[RetrievedChunk] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                RetrievedChunk(
                    id=chunk.id,
                    source=chunk.source,
                    text=chunk.text,
                    score=score,
                )
            )

        return results
