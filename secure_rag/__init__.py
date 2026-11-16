"""Permission-aware RAG: the retriever is the security boundary, not the model."""

from __future__ import annotations

from pathlib import Path

from .corpus import chunk_document, load_corpus
from .embed import TfIdfEmbedder
from .identity import Directory
from .llm import LLM, ScriptedLLM
from .pipeline import NaivePipeline, Pipeline, PostFilterPipeline, SecurePipeline
from .store import VectorStore

ROOT = Path(__file__).resolve().parent.parent

PIPELINES: dict[str, type[Pipeline]] = {
    "naive": NaivePipeline,
    "postfilter": PostFilterPipeline,
    "secure": SecurePipeline,
}


def build_store(corpus_dir: Path = ROOT / "corpus") -> VectorStore:
    chunks = [c for d in load_corpus(corpus_dir) for c in chunk_document(d)]
    store = VectorStore(TfIdfEmbedder())
    store.build(chunks)
    return store


def build_pipeline(
    kind: str = "secure",
    *,
    store: VectorStore | None = None,
    directory: Directory | None = None,
    llm: LLM | None = None,
    k: int = 4,
) -> Pipeline:
    return PIPELINES[kind](
        store=store or build_store(),
        directory=directory or Directory.from_file(ROOT / "users.json"),
        llm=llm or ScriptedLLM(),
        k=k,
    )
