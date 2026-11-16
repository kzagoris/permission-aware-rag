"""Three RAG pipelines over the same index.

* `NaivePipeline`      retrieve top-k, hope the system prompt protects the rest.
* `PostFilterPipeline` retrieve top-k, then drop what the user may not read.
* `SecurePipeline`     filter on permissions inside the search, then rank.

All three share the vector store and the model. The only differences are
where the access decision happens and what gets cached and logged.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .acl import can_read
from .guards import looks_like_instructions
from .identity import Directory, Principal
from .llm import ABSTAIN, GUARD_RULE, LLM, Source
from .store import Hit, VectorStore

BASE_SYSTEM = "You are the internal assistant of Example Corp. Answer from the sources."
CITATION = re.compile(r"\[([a-z0-9-]+#\d+)\]")


@dataclass
class Answer:
    text: str
    retrieved: list[str]
    citations: list[str]
    abstained: bool


@dataclass
class TraceEntry:
    user_id: str
    question_sha: str
    chunk_ids: list[str]
    context: str  # what an observability tool would store


@dataclass
class Pipeline:
    store: VectorStore
    directory: Directory
    llm: LLM
    k: int = 4
    trace: list[TraceEntry] = field(default_factory=list)
    cache: dict[tuple[str, str], Answer] = field(default_factory=dict)

    # --- hooks the three variants override -------------------------------
    def retrieve(self, principal: Principal, question: str) -> list[Hit]:
        raise NotImplementedError

    def system_prompt(self) -> str:
        return BASE_SYSTEM

    def cache_key(self, principal: Principal, question: str) -> tuple[str, str]:
        return ("*", _normalize(question))

    def log_context(self, hits: list[Hit]) -> str:
        return "\n".join(h.chunk.text for h in hits)

    # --- the request path ---------------------------------------------------
    def ask(self, user_id: str, question: str) -> Answer:
        principal = self.directory.resolve(user_id)
        key = self.cache_key(principal, question)
        if key in self.cache:
            return self.cache[key]

        hits = self.retrieve(principal, question)
        sources = [Source(h.chunk.id, h.chunk.title, h.chunk.text) for h in hits]
        text = self.llm.generate(self.system_prompt(), question, sources)

        self.trace.append(
            TraceEntry(
                user_id=principal.user_id,
                question_sha=hashlib.sha256(question.encode()).hexdigest()[:12],
                chunk_ids=[h.chunk.id for h in hits],
                context=self.log_context(hits),
            )
        )
        answer = Answer(
            text=text,
            retrieved=[h.chunk.id for h in hits],
            citations=CITATION.findall(text),
            abstained=text == ABSTAIN,
        )
        self.cache[key] = answer
        return answer


def _normalize(question: str) -> str:
    return " ".join(question.lower().split())


class NaivePipeline(Pipeline):
    """What the candidate described: RAG plus a polite system prompt."""

    def retrieve(self, principal: Principal, question: str) -> list[Hit]:
        return self.store.search(question, self.k)

    def system_prompt(self) -> str:
        return BASE_SYSTEM + " " + GUARD_RULE


class PostFilterPipeline(NaivePipeline):
    """Closer, but top-k is taken before the filter, so results starve."""

    def retrieve(self, principal: Principal, question: str) -> list[Hit]:
        hits = self.store.search(question, self.k)
        return [h for h in hits if can_read(principal, h.chunk)]


class SecurePipeline(Pipeline):
    """Permission filter inside the search, permission-scoped cache, redacted logs."""

    quarantine: bool = True  # drop chunks that address the model (defense in depth)

    def retrieve(self, principal: Principal, question: str) -> list[Hit]:
        return self.store.search(
            question,
            self.k,
            where=lambda c: can_read(principal, c)
            and not (self.quarantine and looks_like_instructions(c)),
        )

    def cache_key(self, principal: Principal, question: str) -> tuple[str, str]:
        return (principal.fingerprint(), _normalize(question))

    def log_context(self, hits: list[Hit]) -> str:
        # Ids and hashes are enough to debug retrieval. The text stays in the
        # document system, behind its own access control.
        return "\n".join(
            f"{h.chunk.id} sha={hashlib.sha256(h.chunk.text.encode()).hexdigest()[:10]}"
            for h in hits
        )
