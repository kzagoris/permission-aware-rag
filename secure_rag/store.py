"""An in-memory vector store with a metadata filter.

`search(..., where=...)` applies the filter *before* ranking, which is what
pre-filtering means in Qdrant, pgvector, Azure AI Search or Pinecone. The
filter shrinks the candidate set; top-k is taken from what remains.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .corpus import Chunk
from .embed import Embedder, Vector, cosine


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._rows: list[tuple[Chunk, Vector]] = []

    def build(self, chunks: list[Chunk]) -> None:
        self.embedder.fit(c.text for c in chunks)
        self._rows = [(c, self.embedder.embed(c.title + " " + c.text)) for c in chunks]

    def search(
        self,
        query: str,
        k: int,
        where: Callable[[Chunk], bool] | None = None,
    ) -> list[Hit]:
        q = self.embedder.embed(query)
        hits = [
            Hit(chunk, cosine(q, vec))
            for chunk, vec in self._rows
            if where is None or where(chunk)
        ]
        hits = [h for h in hits if h.score > 0]
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits[:k]

    def update_acl(self, doc_id: str, groups: frozenset[str]) -> int:
        """Change who may read a document without re-embedding it."""
        changed = 0
        for i, (chunk, vec) in enumerate(self._rows):
            if chunk.doc_id == doc_id:
                self._rows[i] = (replace(chunk, allowed_groups=groups), vec)
                changed += 1
        return changed

    def chunks(self) -> list[Chunk]:
        return [c for c, _ in self._rows]
