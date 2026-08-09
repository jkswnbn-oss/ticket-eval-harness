# ticket-eval-harness

As I've gained traction on how AI can transform my work, I've found that its capability — rising seemingly daily — isn't always in line with its reliability. I support some of the largest ARR logos across multiple verticals, and one slip-up from an AI workflow can have massive implications, none bigger than the loss of trust in me as their TAM.

I wanted a way to use AI on incoming tickets for the fastest possible resolution times without giving up accuracy — and to be able to prove the accuracy rather than assert it.

The harness generates a synthetic enterprise support ticket dataset with gold labels, then runs versioned prompts against it. Each prompt asks the model to do what a frontline engineer does on first touch: assign severity (P1–P4), pick a routing decision, and draft the actual first response to the customer. It's explicitly instructed to read past the customer's tone, threats, and their own guess at the cause — that's the failure mode I care most about, because an angry P4 and a calm P1 both get mishandled by a model that anchors on sentiment. Output is graded by six deterministic checks and two LLM judges, producing a scorecard broken out per grader and per prompt version.

This is a proof of concept, not a product. But building it changed how I understand what it takes to put an LLM workflow into a high-pressure use case.

## Quickstart

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-...
```

The dataset in `data/` is already generated and committed, so you can skip straight to `run.py`. The four stages, in order:

```bash
# 1. Generate (or regenerate/extend) the dataset — optional, already done.
python src/generate.py --n-tickets 150

# 2. Run a model + prompt version over the dataset.
python src/run.py --model claude-haiku-4-5 --prompt-version v1

# 3. Grade that run against gold labels.
python src/grade.py --run-id claude-haiku-4-5__v1__20260807T220000Z

# 4. Aggregate into a scorecard.
python src/scorecard.py claude-haiku-4-5__v1__20260807T220000Z
```

`run_id` defaults to `<model>__<prompt_version>__<utc-timestamp>`, printed at the top of `run.py`'s output — copy it forward into `grade.py` and `scorecard.py`. Every stage that calls the API has a `--dry-run` flag (stub responses, no network, no cost) for smoke-testing plumbing:

```bash
python src/run.py --dry-run --limit 5
python src/grade.py --run-id <run-id> --dry-run --limit 5
```

## How it works

**Generate** (`src/generate.py`) calls the Anthropic API in batches with forced structured output (tool use), steering each batch toward whichever severity / reporter-profile / product-area buckets are under-represented so far, and validates every record against `src/schema.py` before writing `data/tickets.json` and `data/gold_labels.json`.

**Run** (`src/run.py`) loads `data/tickets.json` — never the gold labels — and sends each ticket through a configurable model and prompt version, with bounded async concurrency and retry-with-backoff on transient errors. It writes one JSON record per ticket to `results/<run_id>.jsonl` plus a `.meta.json` sidecar, and is resumable: re-invoking with the same `--run-id` skips tickets that already have a successful (`error: null`) record.

**Grade** (`src/grade.py`) reads a run's `.jsonl` output and `data/gold_labels.json`, runs six deterministic graders and two LLM-as-judge graders over every ticket, and writes one row per `(ticket_id, grader_name)` to `results/<run_id>.grades.jsonl` plus a `.grades.meta.json` sidecar. It never re-calls the model under test, only reads what `run.py` already produced, and is resumable the same way.

**Scorecard** (`src/scorecard.py`) takes one or more run ids, loads each `.grades.jsonl`, and prints a per-grader mean-score/pass-rate table plus a breakdown by prompt version — to the terminal via `rich`, and to a markdown file `results/scorecard__<run-ids-joined>.md`.

## The dataset

150 tickets in `data/tickets.json`, 150 matching gold labels in `data/gold_labels.json` (keyed by ticket id). Current distribution:

| dimension | breakdown |
|---|---|
| severity | P3 61, P2 37, P4 29, P1 23 (target fractions at generation time: 15% P1 / 25% P2 / 40% P3 / 20% P4) |
| product area | search 25, analytics 25, integrations 25, auth 25, content-management 25, ai-assistant 25 |
| reporter profile | precise-technical 74, wrong-diagnosis 30, vague-frustrated 27, multi-issue 10, escalation-threat 9 |
| customer tier | standard 81, gold 43, platinum 26 |
| channel | portal 52, chat 52, email 46 |

`reporter_profile == "wrong-diagnosis"` tickets (20% of the dataset) are the traps: the customer confidently states an incorrect cause, and `gold.true_issue` deliberately diverges from it.

**Ticket fields** (`src/schema.py:Ticket` — what the model under test sees): `id`, `created_at`, `customer_tier`, `product_area`, `channel`, `subject`, `body`, `reporter_profile`.

**Gold label fields** (`src/schema.py:GoldLabel` — never shown to the model under test): `severity`, `true_issue` (one sentence, the actual ground-truth cause), `correct_routing`, `key_facts_needed` (list of facts a good response must acknowledge or request).

## Graders

| grader | type | measures | scoring |
|---|---|---|---|
| `output_parses` | deterministic | does `raw_output` parse as a JSON object | 1.0/0.0, passes iff parses |
| `has_required_fields` | deterministic | parsed object has `severity`, `routing`, `response_draft` | 1.0/0.0, passes iff all three present |
| `valid_severity` | deterministic | `severity` is one of P1–P4 | 1.0/0.0, passes iff a valid enum value |
| `valid_routing` | deterministic | `routing` is one of the four routing literals | 1.0/0.0, passes iff a valid enum value |
| `severity_exact_match` | deterministic | parsed `severity` == `gold_labels[id].severity` | 1.0/0.0, passes iff exact match |
| `routing_exact_match` | deterministic | parsed `routing` == `gold_labels[id].correct_routing` | 1.0/0.0, passes iff exact match |
| `response_quality` | LLM judge | does `response_draft` address this customer's specifics, right tone, no leaked internal jargon | judge scores 1–5 via a `submit_judgment` tool call, normalized to score/5.0 |
| `triage_reasoning` | LLM judge | is the severity/routing decision defensible given `gold.true_issue` and `key_facts_needed` (not just an exact-match check — a reasoned near-miss can score well, a lucky match on the customer's wrong self-diagnosis can score poorly) | judge scores 1–5 via `submit_judgment`, normalized to score/5.0 |

Every grader returns `(score, passed, reason)`, `score` always normalized 0.0–1.0. Deterministic graders pass iff `score == 1.0`. Judge graders pass iff `score >= judge_pass_threshold`, a module-level constant in `src/graders.py` (`JUDGE_PASS_THRESHOLD`), currently **0.6** (3/5). It isn't yet exposed as a CLI flag — changing it means editing that constant — but every grade row denormalizes the threshold it was actually computed against, so historical rows stay meaningful even if the default changes later.

## Prompt versioning

Task prompts live in `PROMPT_VERSIONS` (`src/prompts.py`); judge prompts live in `RESPONSE_QUALITY_JUDGE_PROMPTS` and `TRIAGE_REASONING_JUDGE_PROMPTS` (`src/judge_prompts.py`), versioned independently since the two judges' rubrics evolve on separate timelines. All three dicts are append-only by convention — a version's text is never mutated once added, since results and grade records reference it by string and would silently stop being reproducible otherwise.

`grade.py` denormalizes `prompt_version` from the run's `.meta.json` onto every grade row, and `scorecard.py` breaks its aggregation out by `prompt_version` so different versions can sit side by side in the same table. **Today only `v1` exists** for the task prompt and both judge prompts — the cross-version comparison path is implemented and exercised by `scorecard.py`'s multi-run-id support, but there has been no actual v1-vs-v2 comparison run yet.

## Design notes

- **Prompts and judge prompts are versioned and append-only** so a run or grade record from months ago stays reproducible — you can always look up exactly what text produced a given score, even after the prompt has since been revised.
- **Judge structured output goes through tool use** (`submit_judgment`, forced via `tool_choice`) rather than asking the judge to emit parseable free text — the same pattern `generate.py` uses for dataset generation (`submit_tickets`). It guarantees a valid `score`/`reason` pair instead of adding a text-parsing failure mode on top of the judgment itself.
- **Grade records denormalize `model` and `prompt_version`** (copied from the run's `.meta.json` at grading time) onto every one of the `(ticket_id, grader_name)` rows in `.grades.jsonl`, so `scorecard.py` — or anyone reading the JSONL directly — can group or filter by them without joining back to the run file.
- **`.grades.meta.json`** is the grading run's own sidecar (mirroring `run.py`'s `.meta.json`): judge model, deterministic grader version, per-judge prompt versions, `judge_pass_threshold`, and `started_at` — one canonical record of what a whole grading invocation used, since that doesn't travel if `.grades.jsonl` files from different grading runs are ever concatenated.

## Limitations

- **Synthetic data, not real tickets.** Both the tickets and their gold labels are LLM-generated (`generate.py`, forced tool use, validated against `schema.py`) rather than sourced from actual support history — realistic by construction, not by observation.
- **Single judge model, no ensemble.** Both LLM-as-judge graders use one `--judge-model` call per grading run; there's no self-consistency or multi-model voting to smooth out judge variance.
- **No human agreement baseline.** Nothing in this repo checks the judge rubrics, or the gold labels themselves, against a human support lead's actual judgment — the "ground truth" is itself model-generated.
- **Thin per-slice samples.** 150 tickets split six ways by product area is 25 each; any breakdown finer than the top-line per-grader numbers (e.g. per-product-area accuracy) would be working off a small sample.
- **Cross-version comparison is unexercised.** `scorecard.py` accepts multiple run ids and `prompts.py`/`judge_prompts.py` are built for it, but with only `v1` in existence there's no real prompt-vs-prompt result to point to yet.

## Sample scorecard output

Terminal output from `python src/scorecard.py <run-id>` (`--dry-run`-graded data, so the exact-match numbers reflect the stub's fixed `P3`/`resolve-frontline` output against real gold labels, not a real model's accuracy — shown here for the table shape, not the numbers):

```
                    Per-grader summary (all runs combined)
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ grader               ┃ type          ┃ n  ┃ errors ┃ mean score ┃ pass rate ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ has_required_fields  │ deterministic │ 10 │ 0      │ 1.00       │ 100%      │
│ output_parses        │ deterministic │ 10 │ 0      │ 1.00       │ 100%      │
│ response_quality     │ llm_judge     │ 10 │ 0      │ 0.75       │ 100%      │
│ routing_exact_match  │ deterministic │ 10 │ 0      │ 0.00       │ 0%        │
│ severity_exact_match │ deterministic │ 10 │ 0      │ 0.20       │ 20%       │
│ triage_reasoning     │ llm_judge     │ 10 │ 0      │ 0.75       │ 100%      │
│ valid_routing        │ deterministic │ 10 │ 0      │ 1.00       │ 100%      │
│ valid_severity       │ deterministic │ 10 │ 0      │ 1.00       │ 100%      │
└──────────────────────┴───────────────┴────┴────────┴────────────┴───────────┘
         Breakdown by prompt version
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ grader               ┃ v1 mean ┃ v1 pass% ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ has_required_fields  │ 1.00    │ 100%     │
│ output_parses        │ 1.00    │ 100%     │
│ response_quality     │ 0.75    │ 100%     │
│ routing_exact_match  │ 0.00    │ 0%       │
│ severity_exact_match │ 0.20    │ 20%      │
│ triage_reasoning     │ 0.75    │ 100%     │
│ valid_routing        │ 1.00    │ 100%     │
│ valid_severity       │ 1.00    │ 100%     │
└──────────────────────┴─────────┴──────────┘

written summary: results/scorecard__claude-haiku-4-5__v1__20260807T220000Z.md
```
