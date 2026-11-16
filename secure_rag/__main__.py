"""Command line: ask a question as a user, through a pipeline.

    python -m secure_rag ask --user intern-ana --pipeline naive "What is the CEO's salary?"
    python -m secure_rag compare --user intern-ana "How much does the chief executive take home?"
"""

from __future__ import annotations

import argparse
import textwrap

from . import PIPELINES, build_pipeline, build_store
from .llm import ClaudeLLM, ScriptedLLM


def _print(kind: str, answer) -> None:
    print(f"\n== {kind} ==")
    print("retrieved:", ", ".join(answer.retrieved) or "(nothing)")
    print(textwrap.fill(answer.text, 110, replace_whitespace=False, break_on_hyphens=False))


def main() -> None:
    parser = argparse.ArgumentParser(prog="secure_rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("ask", "compare"):
        p = sub.add_parser(name)
        p.add_argument("question")
        p.add_argument("--user", required=True)
        p.add_argument("--claude", action="store_true", help="use Claude instead of the scripted model")
        if name == "ask":
            p.add_argument("--pipeline", choices=sorted(PIPELINES), default="secure")
    args = parser.parse_args()

    llm = ClaudeLLM() if args.claude else ScriptedLLM()
    store = build_store()
    kinds = [args.pipeline] if args.cmd == "ask" else ["naive", "postfilter", "secure"]
    for kind in kinds:
        pipeline = build_pipeline(kind, store=store, llm=llm)
        _print(kind, pipeline.ask(args.user, args.question))


if __name__ == "__main__":
    main()
