"""Filtering after top-k starves the context. Filtering inside the search does not."""

from __future__ import annotations

QUESTION = "How will I hear about a restructuring that affects my role?"


def test_post_filter_returns_fewer_chunks_than_k(make):
    post = make("postfilter").ask("eng-dimitris", QUESTION)
    secure = make("secure").ask("eng-dimitris", QUESTION)
    # Half of the naive top-4 is the restricted restructuring plan. The
    # post-filter throws it away and keeps what is left; the pre-filter ranks
    # only what the engineer may read.
    assert len(post.retrieved) < len(secure.retrieved)
    assert "pol-payroll-faq#1" in secure.retrieved
    assert "pol-payroll-faq#1" not in post.retrieved


def test_system_prompt_guard_also_blocks_legitimate_questions(make):
    """The keyword guard refuses the engineer's own policy question."""
    post = make("postfilter").ask("eng-dimitris", QUESTION)
    secure = make("secure").ask("eng-dimitris", QUESTION)
    assert "can't share" in post.text
    assert "one-to-one meeting" in secure.text
    assert "pol-org-change#0" in secure.citations


def test_authorized_user_gets_the_answer(make):
    answer = make("secure").ask("hr-maria", "Does the Developer Tools team merge into Platform?")
    assert "two engineering roles close" in answer.text
    assert "exec-restructuring-2027#0" in answer.citations
