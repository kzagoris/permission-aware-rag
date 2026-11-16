"""Filtering after top-k starves the context. Filtering inside the search does not."""

from __future__ import annotations

QUESTION = "When is my salary paid?"


def test_post_filter_returns_fewer_chunks_than_k(make):
    post = make("postfilter").ask("intern-ana", QUESTION)
    secure = make("secure").ask("intern-ana", QUESTION)
    # Half of the naive top-4 is HR data. The post-filter throws it away and
    # keeps what is left; the pre-filter ranks only what the intern may read.
    assert len(post.retrieved) < len(secure.retrieved)
    assert "pol-payroll-faq#1" in secure.retrieved
    assert "pol-payroll-faq#1" not in post.retrieved


def test_system_prompt_guard_also_blocks_legitimate_questions(make):
    """The keyword guard refuses the intern's own payday question."""
    post = make("postfilter").ask("intern-ana", QUESTION)
    secure = make("secure").ask("intern-ana", QUESTION)
    assert "can't share" in post.text
    assert "last working day" in secure.text
    assert "pol-payroll-faq#0" in secure.citations


def test_authorized_user_gets_the_answer(make):
    answer = make("secure").ask("hr-maria", "What is the CEO's base salary for 2026?")
    assert "310,000" in answer.text
    assert answer.citations == ["hr-exec-comp-2026#0"] or "hr-exec-comp-2026#0" in answer.citations
