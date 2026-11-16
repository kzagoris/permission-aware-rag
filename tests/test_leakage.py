"""Leaks that happen outside the prompt: the cache, the logs, stale permissions."""

from __future__ import annotations

import re

CANARY = re.compile(r"CANARY-[A-Z]{2}-[0-9A-F]{4}")
QUESTION = "Does the Developer Tools team merge into Platform?"
FACT = "two engineering roles close"


def test_shared_cache_serves_hr_answer_to_the_contractor(make):
    naive = make("naive")
    naive.ask("hr-maria", QUESTION)
    assert FACT in naive.ask("contractor-sam", QUESTION).text


def test_permission_scoped_cache_does_not(make):
    secure = make("secure")
    assert FACT in secure.ask("hr-maria", QUESTION).text
    assert FACT not in secure.ask("eng-dimitris", QUESTION).text


def test_traces_hold_ids_not_text(make):
    secure = make("secure")
    secure.ask("hr-maria", QUESTION)
    entry = secure.trace[-1]
    assert "exec-restructuring-2027#0" in entry.chunk_ids
    assert not CANARY.search(entry.context)
    assert FACT not in entry.context


def test_naive_traces_copy_restricted_text_into_the_logs(make):
    naive = make("naive")
    naive.ask("eng-dimitris", QUESTION)
    assert FACT in naive.trace[-1].context


def test_revoking_a_group_takes_effect_on_the_next_question(make, directory):
    secure = make("secure")
    assert FACT in secure.ask("hr-maria", QUESTION).text
    directory.set_groups("hr-maria", ["all-staff"])
    secure.cache.clear()  # the fingerprint changed anyway; this proves retrieval, not cache
    assert FACT not in secure.ask("hr-maria", QUESTION).text


def test_revocation_also_misses_the_old_cache_entry(make, directory):
    secure = make("secure")
    secure.ask("hr-maria", QUESTION)
    directory.set_groups("hr-maria", ["all-staff"])
    assert FACT not in secure.ask("hr-maria", QUESTION).text


def test_changing_a_document_acl_needs_no_reindex():
    from secure_rag import build_pipeline, build_store

    store = build_store()
    secure = build_pipeline("secure", store=store)
    assert FACT in secure.ask("hr-maria", QUESTION).text
    store.update_acl("exec-restructuring-2027", frozenset({"exec"}))
    secure.cache.clear()
    assert FACT not in secure.ask("hr-maria", QUESTION).text
    assert FACT in secure.ask("cfo-nikos", QUESTION).text
