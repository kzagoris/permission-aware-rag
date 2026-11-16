"""Load the document corpus and split it into chunks.

Every chunk inherits the access list of its source document. The access list
travels with the chunk into the index, so the retriever can filter on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    allowed_groups: frozenset[str]
    sensitivity: str
    body: str


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str
    allowed_groups: frozenset[str] = field(default_factory=frozenset)
    sensitivity: str = "unknown"


_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def parse_document(raw: str) -> Document:
    match = _FRONTMATTER.match(raw.replace("\r\n", "\n"))
    if not match:
        raise ValueError("document has no frontmatter")
    header, body = match.groups()
    meta: dict[str, str] = {}
    for line in header.splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    groups = frozenset(g.strip() for g in meta.get("allowed_groups", "").split(",") if g.strip())
    return Document(
        id=meta["id"],
        title=meta.get("title", meta["id"]),
        allowed_groups=groups,
        sensitivity=meta.get("sensitivity", "unknown"),
        body=body.strip(),
    )


def load_corpus(directory: Path) -> list[Document]:
    return [parse_document(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.md"))]


def chunk_document(doc: Document) -> list[Chunk]:
    """One chunk per paragraph. Small chunks keep the demo easy to follow."""
    paragraphs = [p.strip() for p in doc.body.split("\n\n") if p.strip()]
    return [
        Chunk(
            id=f"{doc.id}#{i}",
            doc_id=doc.id,
            title=doc.title,
            text=" ".join(p.split()),
            allowed_groups=doc.allowed_groups,
            sensitivity=doc.sensitivity,
        )
        for i, p in enumerate(paragraphs)
    ]
