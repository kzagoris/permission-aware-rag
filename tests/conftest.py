from __future__ import annotations

import pytest

from secure_rag import ROOT, build_pipeline, build_store
from secure_rag.corpus import load_corpus
from secure_rag.identity import Directory


@pytest.fixture(scope="session")
def store():
    return build_store()


@pytest.fixture(scope="session")
def documents():
    return load_corpus(ROOT / "corpus")


@pytest.fixture
def directory():
    return Directory.from_file(ROOT / "users.json")


@pytest.fixture
def make(store, directory):
    def _make(kind: str, **kwargs):
        return build_pipeline(kind, store=kwargs.pop("store", store), directory=directory, **kwargs)

    return _make
