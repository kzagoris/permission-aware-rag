# permission-aware-rag

A small RAG assistant over a fake company's documents, built three ways, with
a test suite that shows which one leaks.

An engineer asks if their team is affected by the restructuring. The vector
search finds the leadership's restructuring plan, and the model answers
correctly: the team merges into Platform and two roles close. The answer is
accurate, grounded and cited. It is also a data breach. This repository turns
that scenario into code and tests.

![Architecture](docs/architecture.svg)

The companion article on Witty Programming walks through every part of it:
[The LLM Is Not Your Security Boundary](https://www.wittyprogramming.dev/articles/permission-aware-rag/).

## The three pipelines

| Pipeline     | Where the access decision happens                      | What goes wrong                                          |
| ------------ | ------------------------------------------------------ | -------------------------------------------------------- |
| `naive`      | In the system prompt ("never reveal restructuring...") | Rephrase the question, or inject, and the data comes out |
| `postfilter` | After top-k: retrieve, then drop forbidden chunks      | Fewer chunks than `k`, sometimes zero                    |
| `secure`     | Inside the search, before ranking                      | Nothing the user can't already read reaches the model    |

All three share the same index and the same model. The only difference is
where the permission check runs, and what the cache and the logs keep.

## Run it

No API key and no dependencies are needed. The embedder is a small TF-IDF
and the default model is a scripted stand-in, so every result is
deterministic.

```bash
git clone https://github.com/kzagoris/permission-aware-rag.git
cd permission-aware-rag
uv run --group dev pytest
```

Ask the same question through all three pipelines:

```bash
uv run python -m secure_rag compare --user eng-dimitris "Does the Developer Tools team merge into Platform?"
```

```text
== naive ==
retrieved: exec-restructuring-2027#0, pol-security#2, wiki-offsite#0, legal-litigation-memo#1
In the first quarter of 2027, the Developer Tools team merges into the Platform team and two engineering roles
close. [exec-restructuring-2027#0]

== postfilter ==
retrieved: pol-security#2, wiki-offsite#0
I don't know. None of the documents you can access answer this.

== secure ==
retrieved: pol-security#2, wiki-offsite#0, wiki-onboarding#0, wiki-offsite#1
I don't know. None of the documents you can access answer this.
```

Run the evaluation:

```bash
uv run python -m secure_rag.evals --pipeline secure
```

To use Claude instead of the scripted model, install the extra and pass
`--claude`. It reads `ANTHROPIC_API_KEY` from the environment.

```bash
uv sync --extra claude
uv run python -m secure_rag compare --claude --user eng-dimitris "Is my team affected by the restructuring?"
```

## What the tests check

| File                          | Claim                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| `tests/test_access_matrix.py` | For every persona and every probe, no forbidden chunk is retrieved and no foreign canary appears        |
| `tests/test_retrieval.py`     | Post-filtering starves the context; the keyword guard also refuses legitimate questions                 |
| `tests/test_injection.py`     | Injection still succeeds, but reaches only the user's own documents; quarantine stops a poisoned page   |
| `tests/test_leakage.py`       | Shared caches and verbose traces leak; revocation and ACL changes apply on the next question            |
| `tests/test_evals.py`         | Recall, citation accuracy, groundedness and abstention stay above the bar in CI                         |

The access tests assert on the retrieved set, not on the model's answer.
Retrieval is deterministic, so the tests are too.

## The corpus

`corpus/` holds 17 short documents with an `allowed_groups` header. Every
restricted document carries a unique canary token, such as `CANARY-EX-4D1E`.
`users.json` stands in for the identity provider:

| User             | Groups                   |
| ---------------- | ------------------------ |
| `intern-ana`     | all-staff, interns       |
| `eng-dimitris`   | all-staff, engineering   |
| `hr-maria`       | all-staff, hr            |
| `legal-eleni`    | all-staff, legal         |
| `cfo-nikos`      | all-staff, finance, exec |
| `contractor-sam` | contractors              |

One document, `draft-untagged.md`, has no access list, so nobody can
retrieve it. Another, `wiki-offsite.md`, contains a prompt injection aimed at
the assistant.

## Limitations

The scripted model is a caricature. It obeys any "ignore previous
instructions" it sees, and its guard matches keywords. A real model resists
far better. The tests still hold with a real model, because none of them
depends on the model behaving well.

The TF-IDF embedder stands in for a real embedding model. The permission
filter works on metadata, so swapping the embedder changes the ranking, not
the access decision.

The injection quarantine in `secure_rag/guards.py` is a regular expression.
It catches the poisoned page in this corpus and will miss a paraphrase. It is
defense in depth, not the boundary.

## License

MIT
