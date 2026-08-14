"""Pluggable embedders: sentence-transformers (default) or pure-numpy hashing TF-IDF."""

from __future__ import annotations

import hashlib
import math
import warnings
from typing import Protocol

import numpy as np

from .text_utils import tokenize


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (len(texts), dim) float32 array of L2-normalized embeddings."""
        ...


class TfidfEmbedder:
    """Stateless hashing TF-IDF (sublinear tf, L2-normalized).

    No model download, no network, fully deterministic — used by tests and as the
    automatic fallback when sentence-transformers is unavailable.
    """

    def __init__(self, dim: int = 4096):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[int, int] = {}
            for tok in tokenize(text):
                digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest, "little") % self.dim
                counts[bucket] = counts.get(bucket, 0) + 1
            for bucket, count in counts.items():
                out[i, bucket] = 1.0 + math.log(count)
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out


class SentenceTransformerEmbedder:
    """Local sentence-transformers model (downloads weights on first use)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        try:
            self.dim = int(self._model.get_embedding_dimension())
        except AttributeError:  # sentence-transformers < 5.x
            self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)


def build_embedder(backend: str = "auto", model_name: str = "BAAI/bge-small-en-v1.5") -> Embedder:
    """`auto` tries sentence-transformers, falls back to TF-IDF with a warning."""
    if backend in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception as e:
            if backend == "sentence-transformers":
                raise
            warnings.warn(
                f"sentence-transformers unavailable ({type(e).__name__}: {e}); "
                "falling back to tfidf embeddings",
                stacklevel=2,
            )
    if backend in ("auto", "tfidf"):
        return TfidfEmbedder()
    raise ValueError(f"unknown embedding backend: {backend!r}")
