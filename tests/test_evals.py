"""Evaluation thresholds run in CI like any other regression test."""

from __future__ import annotations

from secure_rag.evals import evaluate


def test_secure_pipeline_meets_the_bar(make):
    report = evaluate(make("secure"))
    assert report.leaks == 0
    assert report.recall_at_k >= 0.9
    assert report.citation_accuracy >= 0.9
    assert report.groundedness >= 0.9
    assert report.abstention_accuracy >= 0.9


def test_naive_pipeline_fails_the_bar(make):
    report = evaluate(make("naive"))
    assert report.leaks > 0 or report.abstention_accuracy < 0.9
