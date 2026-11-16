"""Groundedness and accuracy evaluation.

Each gold item names the user who asks. The expected answer depends on who
asks: for an engineer, "I don't know" is the correct answer to a question about
the restructuring plan. An eval set without a user column cannot measure that.

Metrics, kept separate on purpose:

* retrieval recall@k   - did the expected document reach the context?
* citation accuracy    - does the answer cite the expected document, and only retrieved chunks?
* groundedness         - is every answer sentence supported by the chunk it cites?
* abstention accuracy  - does the system say "I don't know" exactly when it should?
* answer correctness   - does the answer contain the expected fact?

Run:  python -m secure_rag.evals [--pipeline secure|postfilter|naive]
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import ROOT, build_pipeline
from .embed import tokenize
from .pipeline import Pipeline

SENTENCE_WITH_CITE = re.compile(r"([^\[\]]+?)\s*\[([a-z0-9-]+#\d+)\]")


@dataclass
class Report:
    n: int
    recall_at_k: float
    citation_accuracy: float
    groundedness: float
    abstention_accuracy: float
    correctness: float
    leaks: int

    def rows(self) -> list[tuple[str, str]]:
        pct = lambda x: f"{x:6.1%}"
        return [
            ("items", str(self.n)),
            ("retrieval recall@k", pct(self.recall_at_k)),
            ("citation accuracy", pct(self.citation_accuracy)),
            ("groundedness", pct(self.groundedness)),
            ("abstention accuracy", pct(self.abstention_accuracy)),
            ("answer correctness", pct(self.correctness)),
            ("answers with restricted data", str(self.leaks)),
        ]


def supported(sentence: str, chunk_text: str, threshold: float = 0.8) -> bool:
    terms = set(tokenize(sentence))
    if not terms:
        return True
    return len(terms & set(tokenize(chunk_text))) / len(terms) >= threshold


def evaluate(pipeline: Pipeline, gold_path: Path = ROOT / "evals" / "gold.jsonl") -> Report:
    items = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = {c.id: c.text for c in pipeline.store.chunks()}
    recall = cite = grounded = abstain = correct = 0
    answerable = 0
    claims = claims_ok = 0
    leaks = 0
    for item in items:
        answer = pipeline.ask(item["user"], item["question"])
        should_answer = item["expected_doc"] is not None
        abstain += int(answer.abstained != should_answer)
        for forbidden in item.get("must_not_contain", []):
            leaks += int(forbidden in answer.text)
        if not should_answer:
            correct += int(answer.abstained)
            continue
        answerable += 1
        retrieved_docs = {c.split("#")[0] for c in answer.retrieved}
        recall += int(item["expected_doc"] in retrieved_docs)
        cited_docs = {c.split("#")[0] for c in answer.citations}
        cite += int(item["expected_doc"] in cited_docs and set(answer.citations) <= set(answer.retrieved))
        correct += int(item["expected_fact"].lower() in answer.text.lower())
        for sentence, cid in SENTENCE_WITH_CITE.findall(answer.text):
            claims += 1
            claims_ok += int(cid in texts and supported(sentence, texts[cid]))
    grounded = claims_ok / claims if claims else 1.0
    return Report(
        n=len(items),
        recall_at_k=recall / answerable,
        citation_accuracy=cite / answerable,
        groundedness=grounded,
        abstention_accuracy=abstain / len(items),
        correctness=correct / len(items),
        leaks=leaks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="secure_rag.evals")
    parser.add_argument("--pipeline", default="secure", choices=["naive", "postfilter", "secure"])
    args = parser.parse_args()
    report = evaluate(build_pipeline(args.pipeline))
    width = max(len(k) for k, _ in report.rows())
    print(f"pipeline: {args.pipeline}")
    for key, value in report.rows():
        print(f"  {key:<{width}}  {value}")


if __name__ == "__main__":
    main()
