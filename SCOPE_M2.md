# M2 — Runner

## Goal

Build `src/run.py`: load the 150-ticket dataset (`data/tickets.json`), run each
ticket through a model under test with a configurable prompt, and write one
structured JSONL results file per run. No grading, scoring, or scorecard
logic — that's M3.

## In scope

- **Load the dataset** from `data/tickets.json` (validated against
  `src/schema.py:Ticket`). Gold labels (`data/gold_labels.json`) are **not**
  read by the runner — grading is a separate milestone and must not leak into
  eval input.
- **Configurable model and prompt.** Both are CLI flags (`--model`,
  `--prompt-version`), so the same dataset can be replayed against different
  models or prompt revisions without touching code. Prompt text lives in
  `src/prompts.py`, keyed by version string, so new prompt versions are just
  a new dict entry.
- **One JSONL results file per run**, at
  `results/<run_id>.jsonl`. `run_id` defaults to
  `<model>__<prompt_version>__<utc-timestamp>` (filesystem-safe slug) but can
  be overridden with `--run-id`. A sidecar `results/<run_id>.meta.json`
  records the run's model, prompt version, dataset path, limit, and
  concurrency for later reference.
- **Per-ticket record** (one JSON object per line):
  - `run_id`, `ticket_id`, `model`, `prompt_version`, `called_at` (ISO-8601)
  - `input` — the ticket fields sent to the model (no gold data)
  - `raw_output` — the model's raw text response, unparsed
  - `stop_reason`
  - `usage` — `{input_tokens, output_tokens}` (or `null` on error)
  - `latency_ms`
  - `error` — `null`, or `{type, message}` on failure
- **Resumable.** On start, the runner reads any existing lines in the target
  run's output file, treats tickets with a successful (`error: null`) record
  as done, and skips them. Only pending/failed tickets are (re)attempted.
  Results are appended and flushed line-by-line, so a killed or interrupted
  run can simply be re-invoked with the same `--run-id` to pick up where it
  left off.
- **Bounded concurrency.** An `asyncio.Semaphore` caps in-flight API calls
  (`--concurrency`, default 5, hard-capped at 20 to avoid blowing through
  rate limits). A single writer lock serializes appends to the output file.
- **Retry with backoff.** Rate limit errors (429), connection errors, and
  5xx/overloaded errors are retried with exponential backoff + jitter, up to
  `--max-retries` (default 5). Non-retryable errors (4xx other than 429) are
  recorded immediately as a failed ticket record, not retried.
- **`--limit N`** runs only the first N tickets from the dataset, for smoke
  tests.
- **Stub model mode (`--dry-run`).** A stub call path that returns synthetic
  output with simulated latency and no network access, used to verify
  plumbing (dataset loading, concurrency, resumability, output format)
  without spending API budget. Built and tested first; the real Anthropic API
  path is the same code path with `call_stub` swapped for `call_anthropic`.

## Out of scope (M3)

- Grading/scoring model outputs against gold labels
- Aggregate metrics or a scorecard
- Reading `data/gold_labels.json` from the runner

## Design notes

- Async (`asyncio` + `anthropic.AsyncAnthropic`) rather than threads, so the
  concurrency cap, retry/backoff sleeps, and stub-mode simulated latency all
  compose without extra plumbing.
- Model default follows house convention (`claude-opus-4-8`) but is fully
  overridable — the entire point of the harness is comparing models/prompts.
