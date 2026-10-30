"""Retrieval evaluation: hit@1, hit@k and MRR against a labelled question set.

Run `garage-copilot eval` after changing chunking, embeddings or query expansion, so
retrieval quality is measured rather than eyeballed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from garage_copilot.rag.assistant import expand_query
from garage_copilot.rag.store import VectorStore


@dataclass
class EvalReport:
    n: int
    hit_at_1: float
    hit_at_k: float
    mrr: float
    k: int
    misses: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "hit@1": round(self.hit_at_1, 3),
            f"hit@{self.k}": round(self.hit_at_k, 3),
            "mrr": round(self.mrr, 3),
            "misses": self.misses,
        }


def load_questions(path: Path) -> list[dict[str, str]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evaluate_retrieval(
    store: VectorStore, questions: list[dict[str, str]], k: int = 3
) -> EvalReport:
    hits1 = hitsk = 0
    rr = 0.0
    misses: list[str] = []
    for item in questions:
        results = store.search(expand_query(item["question"]), k=k)
        ranks = [i for i, h in enumerate(results, start=1) if h.chunk.heading == item["section"]]
        if ranks:
            hitsk += 1
            rr += 1 / ranks[0]
            hits1 += ranks[0] == 1
        else:
            misses.append(item["question"])
    n = len(questions)
    return EvalReport(n, hits1 / n, hitsk / n, rr / n, k, misses)
