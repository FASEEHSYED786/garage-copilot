"""Retrieval-augmented question answering over the workshop knowledge base."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from garage_copilot.llm import LLMClient
from garage_copilot.rag.chunking import load_directory
from garage_copilot.rag.embeddings import Embedder, TfidfEmbedder
from garage_copilot.rag.store import SearchHit, VectorStore

SYSTEM_PROMPT = """You are a service adviser at a UK garage. Answer the question using ONLY
the numbered context passages. Cite passages like [1]. If the context does not contain the
answer, say you don't know and suggest who to ask. Keep it under 120 words."""

NO_ANSWER = (
    "I couldn't find that in the workshop knowledge base. Please check with the service manager."
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9£]+")
_STOP = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "do",
    "does",
    "can",
    "i",
    "my",
    "we",
    "our",
    "to",
    "of",
    "for",
    "and",
    "or",
    "in",
    "on",
    "it",
    "what",
    "how",
    "when",
    "who",
    "with",
    "be",
    "if",
    "car",
    "cars",
    "vehicle",
    "get",
    "happens",
    "will",
    "there",
}
# Customers and staff use different words for the same thing. Expanding the query before
# retrieval is a cheap win for sparse (TF-IDF) search; dense embeddings need it less.
SYNONYMS = {
    "loan car": "courtesy car",
    "fill up": "fuel level return",
    "refuel": "fuel level return",
    "hire car": "courtesy car",
    "replacement car": "courtesy car",
    "fine": "fines penalty charges notice",
    "ticket": "fines penalty charges notice",
    "stereo": "head unit audio",
    "radio": "head unit audio",
}


def expand_query(question: str) -> str:
    q = question.lower()
    extra = [v for k, v in SYNONYMS.items() if k in q]
    return f"{question} {' '.join(extra)}" if extra else question


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    hits: list[SearchHit] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return bool(self.hits)


class RagAssistant:
    def __init__(self, store: VectorStore, llm: LLMClient | None = None, top_k: int = 3):
        self.store = store
        self.llm = llm
        self.top_k = top_k

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        only: tuple[str, ...] | None = None,
    ) -> RagAssistant:
        store = VectorStore(embedder or TfidfEmbedder())
        store.index(load_directory(directory, only=only))
        return cls(store, llm)

    def ask(self, question: str) -> Answer:
        hits = self.store.search(expand_query(question), k=self.top_k)
        if not hits:
            return Answer(NO_ANSWER)
        citations = [h.chunk.citation for h in hits]
        if self.llm is None:
            text = self._extractive(question, hits)
            used = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
            cited = [c if i in used else None for i, c in enumerate(citations, start=1)]
            return Answer(text, [c for c in cited if c], hits)
        context = "\n\n".join(
            f"[{i}] ({h.chunk.citation})\n{h.chunk.text}" for i, h in enumerate(hits, start=1)
        )
        text = self.llm.complete(SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {question}")
        return Answer(text, citations, hits)

    @staticmethod
    def _extractive(question: str, hits: list[SearchHit], max_sentences: int = 3) -> str:
        """Offline fallback: pick the sentences that overlap most with the question."""
        q_terms = {w for w in _WORD.findall(expand_query(question).lower()) if w not in _STOP}
        scored: list[tuple[float, int, str, int]] = []
        for rank, hit in enumerate(hits, start=1):
            for pos, sentence in enumerate(_SENTENCE.split(hit.chunk.text.replace("\n", " "))):
                terms = set(_WORD.findall(sentence.lower()))
                overlap = len(q_terms & terms)
                if overlap:
                    scored.append((overlap + 2 * hit.score, -pos, sentence.strip(), rank))
        if not scored:
            best = hits[0]
            first = _SENTENCE.split(best.chunk.text.replace("\n", " "))[0]
            return f"{first} [1]"
        scored.sort(reverse=True)
        # Drop weak matches: keep sentences close to the best overlap.
        floor = max(1.0, scored[0][0] * 0.5)
        picked = [s for s in scored if s[0] >= floor][:max_sentences]
        return " ".join(f"{s} [{rank}]" for _, _, s, rank in picked)
