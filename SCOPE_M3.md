# M3 — Graders and Scorecard

## Goal

Grade an existing run's output against gold labels and produce an aggregate
scorecard. Grading is a separate step from generation: it reads a run's
`results/<run_id>.jsonl` (written by `src/run.py`) plus
`data/gold_labels.json`, and never calls the model under test.

## In scope

### `src/graders.py` — grader functions

Two grader families, kept in clearly separate code paths (separate functions,
separate section of the module, separate `grader_type` tag in output — never
mixed in the same function):

**(a) Deterministic graders** — pure rule-based checks on the structured
fields, no API calls, no ambiguity in the answer:
- `output_parses` — does `raw_output` parse as a JSON object at all
- `has_required_fields` — does the parsed object contain `severity`,
  `routing`, `response_draft`
- `valid_severity` — is `severity` one of `P1`/`P2`/`P3`/`P4`
- `valid_routing` — is `routing` one of the four valid routing literals
- `severity_exact_match` — does parsed `severity` equal
  `gold_labels[ticket_id].severity`
- `routing_exact_match` — does parsed `routing` equal
  `gold_labels[ticket_id].correct_routing`

Each downstream deterministic grader that depends on parsing succeeding
(everything after `output_parses`) scores 0 / fails closed with a reason
noting the parse failure, rather than raising, if `raw_output` doesn't parse.

**(b) LLM-as-judge graders** — for the subjective parts, via the Anthropic
API (`anthropic` is already a dependency, no new package):
- `response_quality` — judges the `response_draft` text: does it actually
  address this customer's specific situation, right tone, no leaked
  internal jargon (severity/routing labels) to the customer
- `triage_reasoning` — judges whether the model's severity/routing decision
  is *defensible*, using `gold_labels[ticket_id].true_issue` and
  `key_facts_needed` as the reference for what was actually going on. This
  is deliberately different from `severity_exact_match`/`routing_exact_match`:
  a model can reason soundly and still land one severity notch off, or vice
  versa parrot the customer's wrong self-diagnosis into a lucky exact match.

Judge prompts are versioned the same way `src/prompts.py` versions task
prompts — one dict per judge (not one shared dict), since the two judges'
prompts evolve independently:

```python
# src/judge_prompts.py
RESPONSE_QUALITY_JUDGE_PROMPTS: dict[str, str] = {"v1": "..."}
TRIAGE_REASONING_JUDGE_PROMPTS: dict[str, str] = {"v1": "..."}
```

Every grader — deterministic or judge — returns a `GraderResult(score, passed,
reason)`. No bare numbers ever get written or printed; `reason` is required
and non-empty.

- `score`: float, normalized to **0.0–1.0** for every grader (deterministic
  graders are 1.0/0.0; judges self-report on a 1–5 rubric internally, stored
  normalized). Normalizing lets the scorecard compute a per-grader mean
  without caring whether the grader is deterministic or a judge.
- `passed`: bool — each grader defines its own pass threshold internally
  (deterministic: score == 1.0; judges: score >= 0.6, i.e. 3/5). Stored
  explicitly per-record rather than recomputed at scorecard time, so the
  threshold is pinned to what actually ran.

### `src/grade.py` — grading CLI

Reads `results/<run_id>.jsonl` + its `.meta.json` + `data/gold_labels.json`,
runs every grader over every ticket record, writes
**`results/<run_id>.grades.jsonl`** (one JSON object per `(ticket_id,
grader_name)` pair — i.e. multiple lines per ticket, one per grader) plus a
`results/<run_id>.grades.meta.json` sidecar (grader versions, judge model,
started_at — mirrors the runner's meta sidecar).

Grade record schema (confirmed):

```jsonc
{
  "run_id": "...",            // the run this grade is for
  "ticket_id": "TKT-0001",
  "model": "claude-haiku-4-5",     // denormalized from the run's meta.json
  "prompt_version": "v1",          // denormalized from the run's meta.json
  "grader_name": "severity_exact_match",   // or "response_quality", etc.
  "grader_type": "deterministic",          // | "llm_judge"
  "grader_version": "det-v1",              // fixed string for det. graders;
                                            // the judge-prompt version key
                                            // (e.g. "v1") for judge graders
  "judge_model": null,        // model used for the judge call; null for
                               // deterministic graders
  "judge_pass_threshold": null,  // the score cutoff `passed` was computed
                                  // against (currently 0.6, i.e. 3/5); null
                                  // for deterministic graders, whose pass
                                  // criterion is exact (score == 1.0), not
                                  // a numeric cutoff
  "score": 1.0,                // normalized 0.0-1.0
  "passed": true,
  "reason": "severity P1 matches gold P1",
  "graded_at": "2026-08-05T...",
  "error": null                // | {type, message} — e.g. judge API failure
}
```

Why this shape:
- One row per `(ticket_id, grader_name)` rather than one wide row per ticket
  with a column per grader — adding a new grader later is an additive change
  (new rows), not a schema migration (new columns on every existing row).
- `model` and `prompt_version` are denormalized onto every grade record
  (copied from the run's `.meta.json` at grading time) so the scorecard step
  — and anyone just reading the `.grades.jsonl` file directly — can group/filter
  by prompt version without a join back to the run file. `grade.py` reads them
  once from `.meta.json` and stamps every row it writes for that run.
- Grading is resumable the same way the runner is: `grade.py` skips
  `(ticket_id, grader_name)` pairs that already have an `error: null` row in
  the target `.grades.jsonl`, so a killed judge-grading run can be re-invoked
  safely without re-spending API budget on graders that already succeeded.
- `judge_pass_threshold` is denormalized onto every row for the same reason
  as `prompt_version`: a `passed` bool is meaningless without the cutoff that
  produced it, and `.grades.meta.json` (which also records it) doesn't travel
  when `.grades.jsonl` files from different grading runs get concatenated or
  read independently.

Flags: `--run-id` (required — which run to grade), `--limit` (grade only the
first N tickets), `--dry-run` (stub judge calls, same spirit as the runner's
`--dry-run`, for verifying plumbing without spending API budget), plus
`--concurrency`/`--max-retries` mirroring the runner since judge calls are
real API calls with the same rate-limit/backoff concerns.

### `src/scorecard.py` — aggregation CLI

Takes one or more `run_id`s, loads each run's `.grades.jsonl` + `.meta.json`,
and prints:
- per-grader mean score and pass rate, across all graded tickets
- the same breakdown split by `prompt_version` (reads each run's
  `prompt_version` from its `.meta.json`), so v1 vs v2 (once a v2 prompt
  exists) sit side by side

Terminal output via `rich` (already a dependency — no new deps). Also writes
a plain-text/markdown summary file, `results/scorecard__<run-ids-joined>.md`.

`--limit` and `--dry-run` apply the same way they do to `grade.py`/`run.py`
(dry-run scorecards over dry-run-graded data, for smoke-testing the
aggregation logic itself).

## Out of scope

- Re-running or modifying the model-under-test path (`src/run.py` is
  untouched except perhaps a doc pointer)
- Any change to the existing run-record JSONL schema
- Web UI / dashboard
- New dependencies beyond `anthropic`, `pydantic`, `rich` (already in
  `pyproject.toml`)

## Design notes

- Grading never re-runs the model; it only reads a run's already-written
  JSONL. If `raw_output` is garbage, that's data for `output_parses` to
  score, not something to route around.
- Deterministic and judge graders live in the same `src/graders.py` module
  but under a clear `# --- deterministic ---` / `# --- llm-as-judge ---`
  section split, with distinct function signatures (deterministic graders
  never take a client/model argument; judge graders always do) so the two
  categories can't accidentally blur.
