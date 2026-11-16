"""Who is asking.

The principal is resolved on the server from the authenticated user id. The
client never sends its own groups or role: a claimed role is just more user
input, and user input is not an access decision.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Principal:
    user_id: str
    groups: frozenset[str]

    def fingerprint(self) -> str:
        """Stable id of the permission set. Caches are keyed by it."""
        raw = ",".join(sorted(self.groups))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class UnknownUser(Exception):
    pass


class Directory:
    """Stands in for the identity provider (Entra ID, Okta, LDAP...).

    It is read on every request, so a revoked group takes effect on the next
    question rather than at the next re-index.
    """

    def __init__(self, users: dict[str, list[str]]) -> None:
        self._users = {u: frozenset(g) for u, g in users.items()}

    @classmethod
    def from_file(cls, path: Path) -> "Directory":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def resolve(self, user_id: str) -> Principal:
        if user_id not in self._users:
            raise UnknownUser(user_id)
        return Principal(user_id, self._users[user_id])

    def set_groups(self, user_id: str, groups: list[str]) -> None:
        self._users[user_id] = frozenset(groups)

    def users(self) -> list[str]:
        return sorted(self._users)
