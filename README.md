# ticket-eval-harness

Eval harness for grading LLM performance on enterprise support ticket triage,
severity classification, and first-response drafting.

**Status:** M1 (dataset), M2 (runner), and M3 (graders + scorecard) done —
see `SCOPE_M2.md` and `SCOPE_M3.md`.

## Setup

```bash
cd support-eval-harness
pip install -e .
```

## Regenerating / extending the dataset

`data/tickets.json` and `data/gold_labels.json` (150 tickets) were authored
directly rather than by running `generate.py` end to end, since this
environment had no `ANTHROPIC_API_KEY` available. `generate.py` is fully
functional for regenerating or extending the dataset with a real key:

```bash
export ANTHROPIC_API_KEY=sk-...
python src/generate.py --n-tickets 150
```

It batches requests to the Anthropic API, forces structured output via tool
use, steers each batch toward under-represented severity / reporter-profile /
product-area buckets, and validates every record against `src/schema.py`
before writing it out.

## Dataset

- `data/tickets.json` — what the model under test sees (no gold labels).
- `data/gold_labels.json` — ground truth, keyed by ticket id.

See `src/schema.py` for the full schema and field semantics.

## Running the eval

```bash
# Smoke-test plumbing (no network calls, no API key needed):
python src/run.py --dry-run --limit 5

# Real run:
export ANTHROPIC_API_KEY=sk-...
python src/run.py --model claude-opus-4-8 --prompt-version v1

# Resume an interrupted or partially-failed run — same --run-id, already
# completed tickets are skipped, only pending/failed ones are retried:
python src/run.py --run-id <run-id-from-above>
```

Each run writes `results/<run_id>.jsonl` (one JSON record per ticket: raw
model output, latency, token counts, errors) plus a `results/<run_id>.meta.json`
sidecar. Prompts are versioned in `src/prompts.py`. See `SCOPE_M2.md` for the
full design (resumability, concurrency, retry behavior).

## Grading a run

Grading is a separate step from generation — it reads an existing run's
JSONL, never calls the model under test:

```bash
# Smoke-test plumbing (stub judge scores, no network calls):
python src/grade.py --run-id <run-id> --dry-run --limit 5

# Real grading (needs ANTHROPIC_API_KEY for the LLM-judge graders):
export ANTHROPIC_API_KEY=sk-...
python src/grade.py --run-id <run-id>
```

Six deterministic graders (`output_parses`, `has_required_fields`,
`valid_severity`, `valid_routing`, `severity_exact_match`,
`routing_exact_match`) plus two LLM-as-judge graders (`response_quality`,
`triage_reasoning`, judged via the Anthropic API against
`data/gold_labels.json`) run over every ticket. Output is
`results/<run_id>.grades.jsonl` — one row per `(ticket_id, grader_name)`,
each with a `score` (0.0-1.0), `passed` bool, and a required `reason`
string. Resumable the same way `run.py` is. See `src/graders.py` and
`src/judge_prompts.py` (versioned the same way `src/prompts.py` versions
task prompts).

## Scorecard

```bash
# One run:
python src/scorecard.py <run-id>

# Compare prompt versions side by side (grade each run first):
python src/scorecard.py <run-id-v1> <run-id-v2>
```

Prints a per-grader mean-score/pass-rate table and a breakdown-by-prompt-
version table to the terminal, and writes the same as a markdown summary
file (`results/scorecard__<run-ids>.md`). See `SCOPE_M3.md` for the full
design, including the grade-record JSONL schema.
