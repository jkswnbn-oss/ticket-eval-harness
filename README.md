# support-eval-harness

Eval harness for grading LLM performance on enterprise support ticket triage,
severity classification, and first-response drafting.

**Status:** M1 in progress (dataset). Runner, graders, and the full writeup
land in later milestones — see the project scope doc for the plan.

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
