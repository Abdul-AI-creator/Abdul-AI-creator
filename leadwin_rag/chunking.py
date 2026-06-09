"""Text loading and chunking helpers for the RAG knowledge base."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .models import KnowledgeChunk


SUPPORTED_TEXT_SUFFIXES = {".md", ".txt", ".rst"}
SUPPORTED_TABLE_SUFFIXES = {".csv"}


def discover_files(paths: Iterable[Path]) -> list[Path]:
    """Return supported files from files or directories."""

    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            for suffix in SUPPORTED_TEXT_SUFFIXES | SUPPORTED_TABLE_SUFFIXES:
                files.extend(path.rglob(f"*{suffix}"))
        elif path.suffix.lower() in SUPPORTED_TEXT_SUFFIXES | SUPPORTED_TABLE_SUFFIXES:
            files.append(path)

    return sorted(set(files))


def load_documents(paths: Iterable[Path]) -> list[tuple[str, str]]:
    """Load supported files into ``(source, text)`` tuples."""

    documents: list[tuple[str, str]] = []
    for path in discover_files(paths):
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_TEXT_SUFFIXES:
            documents.append((str(path), path.read_text(encoding="utf-8")))
        elif suffix in SUPPORTED_TABLE_SUFFIXES:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for idx, row in enumerate(reader, start=1):
                    values = [f"{key}: {value}" for key, value in row.items() if value]
                    if values:
                        documents.append((f"{path}#row-{idx}", ". ".join(values)))

    return documents


def chunk_text(source: str, text: str, chunk_size: int = 900, overlap: int = 120) -> list[KnowledgeChunk]:
    """Split text into overlapping word chunks for retrieval."""

    words = text.split()
    if not words:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be zero or greater")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[KnowledgeChunk] = []
    start = 0
    chunk_number = 1
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(
            KnowledgeChunk(
                id=f"{source}:{chunk_number}",
                source=source,
                text=" ".join(chunk_words),
            )
        )
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
        chunk_number += 1

    return chunks


def chunk_documents(
    documents: Iterable[tuple[str, str]],
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[KnowledgeChunk]:
    """Chunk all loaded documents."""

    chunks: list[KnowledgeChunk] = []
    for source, text in documents:
        chunks.extend(chunk_text(source, text, chunk_size=chunk_size, overlap=overlap))
    return chunks
