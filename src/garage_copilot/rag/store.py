"""A small in-memory vector store. Enough for a workshop's knowledge base (hundreds of
chunks); swap for pgvector or a hosted store if it grows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from garage_copilot.rag.chunking import Chunk
from garage_copilot.rag.embeddings import Embedder


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, embedder: Embedder):
        self._embedder = embedder
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Nothing to index")
        texts = [self._index_text(c) for c in chunks]
        self._embedder.fit(texts)
        self._matrix = self._embedder.embed(texts)
        self._chunks = list(chunks)

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> list[SearchHit]:
        if self._matrix is None:
            raise RuntimeError("Store is empty: call index() first")
        query_vec = self._embedder.embed([query])[0]
        scores = self._matrix @ query_vec
        order = np.argsort(-scores)[:k]
        return [
            SearchHit(self._chunks[i], float(scores[i])) for i in order if scores[i] >= min_score
        ]

    def __len__(self) -> int:
        return len(self._chunks)

    @staticmethod
    def _index_text(chunk: Chunk) -> str:
        # Headings carry a lot of meaning in short docs, so they are weighted in.
        return f"{chunk.doc_title}. {chunk.heading}. {chunk.heading}. {chunk.text}"
