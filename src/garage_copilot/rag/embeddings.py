"""Embedding backends.

`TfidfEmbedder` runs fully offline and is the default. `OpenAIEmbedder` is used when an
OpenAI key is configured. Both return L2-normalised vectors so cosine similarity is a
dot product.
"""

from __future__ import annotations

from typing import Protocol

import httpx
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# Words that appear in almost every workshop question and carry no signal.
DOMAIN_STOP_WORDS = frozenset({"car", "cars", "vehicle", "need", "needs", "new", "work"})


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> None: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class TfidfEmbedder:
    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words=sorted(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS),
            min_df=1,
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embed()")
        return _normalise(self._vectorizer.transform(texts).toarray())


class OpenAIEmbedder:
    URL = "https://api.openai.com/v1/embeddings"

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        client: httpx.Client | None = None,
    ):
        self._client = client or httpx.Client(timeout=30.0)
        self._api_key = api_key
        self._model = model

    def fit(self, texts: list[str]) -> None:  # stateless
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.post(
            self.URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": texts},
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return _normalise(np.array([d["embedding"] for d in data], dtype=float))
