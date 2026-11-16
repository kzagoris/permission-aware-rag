"""The access decision. Deny by default."""

from __future__ import annotations

from .corpus import Chunk
from .identity import Principal


def can_read(principal: Principal, chunk: Chunk) -> bool:
    # A chunk without an access list is visible to nobody. Untagged content
    # is a mistake in the ingestion pipeline, not public content.
    if not chunk.allowed_groups:
        return False
    return not chunk.allowed_groups.isdisjoint(principal.groups)
