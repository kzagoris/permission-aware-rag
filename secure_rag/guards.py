"""Defense in depth around the boundary. None of these is the boundary.

`looks_like_instructions` flags chunks that address the model instead of the
reader. Flagged chunks are quarantined: kept out of the context and sent to a
human for review. It is a heuristic; it will miss paraphrases. It exists to
shrink the attack surface, while the permission filter limits the damage of
whatever gets through.
"""

from __future__ import annotations

import re

from .corpus import Chunk

_PATTERNS = [
    r"ignore (all )?(previous|prior|above) (instructions|rules)",
    r"\b(note|message|instruction)s? (for|to) (the )?(ai|assistant|model|llm)",
    r"\bsystem prompt\b",
    r"\b(reveal|print|dump|include) (every|all) (source|document)s?\b",
]
_INSTRUCTIONS = re.compile("|".join(_PATTERNS), re.IGNORECASE)


def looks_like_instructions(chunk: Chunk) -> bool:
    return bool(_INSTRUCTIONS.search(chunk.text))
