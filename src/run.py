"""Runner for the ticket eval harness.

Loads the ticket dataset, sends each ticket through a model + prompt version,
and writes one structured JSONL results file per run. Resumable: re-invoking
with the same --run-id skips tickets that already have a successful result.

Grading, scoring, and the scorecard are out of scope here (M3) — this only
produces raw model output alongside latency/token/error bookkeeping. Gold
labels are deliberately never read by this script.

Usage:
    # Smoke-test plumbing with no network calls:
    python src/run.py --dry-run --limit 5

    # Real run:
    export ANTHROPIC_API_KEY=sk-...
    python src/run.py --model claude-opus-4-8 --prompt-version v1

    # Resume an interrupted/failed run:
    python src/run.py --run-id claude-opus-4-8__v1__20260722T120000Z
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import DEFAULT_PROMPT_VERSION, PROMPT_VERSIONS, build_messages  # noqa: E402
from schema import TicketDataset  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "tickets.json"
DEFAULT_OUT_DIR = REPO_ROOT / "results"
DEFAULT_MODEL = "claude-haiku-4-5"
MAX_CONCURRENCY = 20


@dataclass
class RunConfig:
    dataset: Path
    out_dir: Path
    run_id: str
    model: str
    prompt_version: str
    limit: int | None
    concurrency: int
    max_retries: int
    max_tokens: int
    dry_run: bool


@dataclass
class ModelCallResult:
    raw_output: str
    stop_reason: str | None
    usage: dict[str, int]


@dataclass
class RunStats:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def default_run_id(model: str, prompt_version: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{_slug(model)}__{_slug(prompt_version)}__{ts}"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    dataset = TicketDataset.model_validate(raw)
    return [json.loads(t.model_dump_json()) for t in dataset.root]


def load_completed_ticket_ids(out_path: Path) -> set[str]:
    """Ticket ids that already have a successful (error: null) record in out_path."""
    if not out_path.exists():
        return set()
    completed: set[str] = set()
    with out_path.open("r") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"  warning: {out_path.name}:{line_no} is not valid JSON "
                    "(likely a partial write from an interrupted run) — ignoring, "
                    "will retry that ticket",
                    file=sys.stderr,
                )
                continue
            if record.get("error") is None and record.get("ticket_id"):
                completed.add(record["ticket_id"])
    return completed


def _is_retryable(exc: Exception) -> bool:
    try:
        import anthropic
    except ImportError:
        return False
    # Most-specific-first: RateLimitError is a subclass of APIStatusError.
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500
    return False


async def call_with_retry(call_fn: Any, max_retries: int) -> ModelCallResult:
    attempt = 0
    while True:
        try:
            return await call_fn()
        except Exception as exc:
            if attempt >= max_retries or not _is_retryable(exc):
                raise
            delay = min(2.0 * (2**attempt), 30.0) + random.uniform(0, 1.0)
            attempt += 1
            await asyncio.sleep(delay)


async def call_stub(ticket: dict[str, Any], prompt_version: str, model: str) -> ModelCallResult:
    """No-network stand-in for the real API call — used by --dry-run."""
    _, user_content = build_messages(ticket, prompt_version)  # exercises the real prompt path
    await asyncio.sleep(random.uniform(0.02, 0.08))
    response_draft = f"[stub:{model}/{prompt_version}] Acknowledged ticket {ticket['id']}."
    payload = {
        "severity": "P3",
        "routing": "resolve-frontline",
        "response_draft": response_draft,
    }
    return ModelCallResult(
        raw_output=json.dumps(payload),
        stop_reason="end_turn",
        usage={
            "input_tokens": len(user_content.split()),
            "output_tokens": len(response_draft.split()),
        },
    )


async def call_anthropic(
    client: Any, ticket: dict[str, Any], prompt_version: str, model: str, max_tokens: int
) -> ModelCallResult:
    system, user_content = build_messages(ticket, prompt_version)
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return ModelCallResult(raw_output=text, stop_reason=response.stop_reason, usage=usage)


async def process_ticket(
    ticket: dict[str, Any],
    cfg: RunConfig,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    out_fh: Any,
    client: Any,
    stats: RunStats,
) -> None:
    async with semaphore:
        started = time.monotonic()
        called_at = datetime.now(timezone.utc).isoformat()
        error: dict[str, str] | None = None
        result: ModelCallResult | None = None

        try:
            if cfg.dry_run:
                call_fn = lambda: call_stub(ticket, cfg.prompt_version, cfg.model)  # noqa: E731
            else:
                call_fn = lambda: call_anthropic(  # noqa: E731
                    client, ticket, cfg.prompt_version, cfg.model, cfg.max_tokens
                )
            result = await call_with_retry(call_fn, cfg.max_retries)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}

        latency_ms = round((time.monotonic() - started) * 1000, 1)

        record = {
            "run_id": cfg.run_id,
            "ticket_id": ticket["id"],
            "model": cfg.model,
            "prompt_version": cfg.prompt_version,
            "called_at": called_at,
            "latency_ms": latency_ms,
            "input": {k: v for k, v in ticket.items() if k != "id"},
            "raw_output": result.raw_output if result else None,
            "stop_reason": result.stop_reason if result else None,
            "usage": result.usage if result else None,
            "error": error,
        }

        async with write_lock:
            out_fh.write(json.dumps(record) + "\n")
            out_fh.flush()

        stats.attempted += 1
        if error is None:
            stats.succeeded += 1
            print(f"  ok    {ticket['id']}  {latency_ms:>8.1f}ms")
        else:
            stats.failed += 1
            print(f"  FAIL  {ticket['id']}  {error['type']}: {error['message']}", file=sys.stderr)


async def run(cfg: RunConfig) -> None:
    tickets = load_dataset(cfg.dataset)
    if cfg.limit is not None:
        tickets = tickets[: cfg.limit]

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / f"{cfg.run_id}.jsonl"
    meta_path = cfg.out_dir / f"{cfg.run_id}.meta.json"

    completed = load_completed_ticket_ids(out_path)
    pending = [t for t in tickets if t["id"] not in completed]

    print(f"run_id={cfg.run_id}")
    print(f"model={cfg.model} prompt_version={cfg.prompt_version} dry_run={cfg.dry_run}")
    print(
        f"dataset tickets: {len(tickets)}  already completed: {len(completed)}  "
        f"to run: {len(pending)}"
    )
    print(f"output: {out_path}")

    meta_path.write_text(
        json.dumps(
            {
                "run_id": cfg.run_id,
                "model": cfg.model,
                "prompt_version": cfg.prompt_version,
                "dataset": str(cfg.dataset),
                "limit": cfg.limit,
                "concurrency": cfg.concurrency,
                "dry_run": cfg.dry_run,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )

    if not pending:
        print("Nothing to do — all requested tickets already have a completed result.")
        return

    client = None
    if not cfg.dry_run:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it, or pass --dry-run to "
                "smoke-test the runner without calling the API."
            )
        client = anthropic.AsyncAnthropic(api_key=api_key)

    semaphore = asyncio.Semaphore(cfg.concurrency)
    write_lock = asyncio.Lock()
    stats = RunStats()

    try:
        with out_path.open("a") as out_fh:
            tasks = [
                process_ticket(ticket, cfg, semaphore, write_lock, out_fh, client, stats)
                for ticket in pending
            ]
            await asyncio.gather(*tasks)
    finally:
        if client is not None:
            await client.close()

    print(
        f"\ndone: {stats.succeeded} succeeded, {stats.failed} failed "
        f"(of {len(pending)} attempted this run; {len(completed)} were already done "
        f"before this run started)"
    )
    if stats.failed:
        print(f"Re-run with the same --run-id {cfg.run_id!r} to retry the failed tickets.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Reuse a previous run's id to resume it. Defaults to "
        "<model>__<prompt_version>__<utc-timestamp>.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        choices=sorted(PROMPT_VERSIONS),
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N tickets.")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a stub model call (no network, no cost) instead of the real "
        "Anthropic API. For verifying plumbing/resumability.",
    )
    args = parser.parse_args()

    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.concurrency > MAX_CONCURRENCY:
        print(
            f"warning: clamping --concurrency {args.concurrency} down to {MAX_CONCURRENCY}",
            file=sys.stderr,
        )
        args.concurrency = MAX_CONCURRENCY

    run_id = args.run_id or default_run_id(args.model, args.prompt_version)

    cfg = RunConfig(
        dataset=args.dataset,
        out_dir=args.out_dir,
        run_id=run_id,
        model=args.model,
        prompt_version=args.prompt_version,
        limit=args.limit,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
    )

    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
