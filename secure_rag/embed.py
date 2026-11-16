"""A tiny TF-IDF embedder, so the demo runs offline and deterministically.

A real system uses a neural embedding model. Nothing in the access-control
logic depends on that choice: the filter runs on metadata, not on vectors.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Protocol

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i in is it its
    me my of on or our per please the their this to us was we what when where
    which who will with you your all any every""".split()
)

Vector = dict[str, float]


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class Embedder(Protocol):
    def fit(self, texts: Iterable[str]) -> None: ...
    def embed(self, text: str) -> Vector: ...


class TfIdfEmbedder:
    def __init__(self) -> None:
        self.idf: dict[str, float] = {}

    def fit(self, texts: Iterable[str]) -> None:
        docs = [set(tokenize(t)) for t in texts]
        n = len(docs)
        df = Counter(term for d in docs for term in d)
        self.idf = {term: math.log((1 + n) / (1 + c)) + 1 for term, c in df.items()}

    def embed(self, text: str) -> Vector:
        tf = Counter(tokenize(text))
        vec = {t: c * self.idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}


def cosine(a: Vector, b: Vector) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())
