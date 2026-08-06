"""Grader runner for the ticket eval harness.

Reads an existing run's JSONL output (written by src/run.py) plus
data/gold_labels.json, runs every grader — deterministic and LLM-as-judge —
over each ticket record, and writes one grade record per
(ticket_id, grader_name) to results/<run_id>.grades.jsonl.

Grading is a separate step from generation: this script never calls the
model under test, only reads its already-written output. See src/graders.py
for the grader implementations and SCOPE_M3.md for the grade-record schema.

Usage:
    # Smoke-test plumbing with no network calls (stub judge scores):
    python src/grade.py --run-id <run_id> --dry-run --limit 5

    # Real grading run (needs ANTHROPIC_API_KEY for the LLM-judge graders;
    # deterministic graders never need it):
    export ANTHROPIC_API_KEY=sk-...
    python src/grade.py --run-id <run_id>

    # Resume an interrupted/failed grading run:
    python src/grade.py --run-id <run_id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graders import (  # noqa: E402
    DETERMINISTIC_GRADER_VERSION,
    DETERMINISTIC_GRADERS,
    JUDGE_PASS_THRESHOLD,
    LLM_JUDGE_GRADER_VERSIONS,
    LLM_JUDGE_GRADERS,
    GraderResult,
    GradingContext,
)
from schema import GoldLabel, GoldLabelSet  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = REPO_ROOT / "data" / "gold_labels.json"
DEFAULT_OUT_DIR = REPO_ROOT / "results"
DEFAULT_JUDGE_MODEL = "claude-opus-4-8"
MAX_CONCURRENCY = 20


@dataclass
class GradeConfig:
    run_id: str
    out_dir: Path
    gold_path: Path
    judge_model: str
    limit: int | None
    concurrency: int
    max_retries: int
    dry_run: bool


@dataclass
class GradeStats:
    graded: int = 0
    skipped: int = 0
    errored: int = 0


def load_gold_labels(path: Path) -> dict[str, GoldLabel]:
    raw = json.loads(path.read_text())
    return GoldLabelSet.model_validate(raw).root


def load_latest_run_records(run_path: Path) -> dict[str, dict[str, Any]]:
    """One record per ticket_id: the successful (error: null) record if one
    exists, else the most recent attempt. Mirrors run.py's own notion of
    "the result for this ticket" for a resumed/retried run file.
    """
    latest: dict[str, dict[str, Any]] = {}
    succeeded: set[str] = set()
    with run_path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ticket_id = record.get("ticket_id")
            if not ticket_id or ticket_id in succeeded:
                continue
            latest[ticket_id] = record
            if record.get("error") is None:
                succeeded.add(ticket_id)
    return latest


def load_completed_grade_keys(out_path: Path) -> set[tuple[str, str]]:
    """(ticket_id, grader_name) pairs that already have an error:null row."""
    if not out_path.exists():
        return set()
    completed: set[tuple[str, str]] = set()
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
                    "will retry that pair",
                    file=sys.stderr,
                )
                continue
            if record.get("error") is None and record.get("ticket_id") and record.get("grader_name"):
                completed.add((record["ticket_id"], record["grader_name"]))
    return completed


def _is_retryable(exc: Exception) -> bool:
    try:
        import anthropic
    except ImportError:
        return False
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500
    return False


async def call_with_retry(call_fn: Any, max_retries: int) -> GraderResult:
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


async def judge_stub(grader_name: str, ctx: GradingContext) -> GraderResult:
    """No-network stand-in for a real judge call — used by --dry-run."""
    await asyncio.sleep(random.uniform(0.02, 0.08))
    has_signal = ctx.parsed is not None
    return GraderResult(
        score=0.75 if has_signal else 0.0,
        passed=has_signal,
        reason=f"[stub:{grader_name}] no model call made (--dry-run)",
    )


def build_grade_record(
    *,
    run_id: str,
    model: str,
    prompt_version: str,
    ticket_id: str,
    grader_name: str,
    grader_type: str,
    grader_version: str,
    judge_model: str | None,
    judge_pass_threshold: float | None,
    result: GraderResult | None,
    error: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "ticket_id": ticket_id,
        "model": model,
        "prompt_version": prompt_version,
        "grader_name": grader_name,
        "grader_type": grader_type,
        "grader_version": grader_version,
        "judge_model": judge_model,
        "judge_pass_threshold": judge_pass_threshold,
        "score": result.score if result else None,
        "passed": result.passed if result else None,
        "reason": result.reason if result else None,
        "graded_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }


async def grade_ticket(
    ticket_id: str,
    run_record: dict[str, Any],
    gold: GoldLabel,
    cfg: GradeConfig,
    completed: set[tuple[str, str]],
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    out_fh: Any,
    client: Any,
    stats: GradeStats,
) -> None:
    ctx = GradingContext(
        ticket_id=ticket_id,
        raw_output=run_record.get("raw_output"),
        ticket_input=run_record.get("input") or {},
        gold=gold,
    )
    run_id = run_record["run_id"]
    model = run_record["model"]
    prompt_version = run_record["prompt_version"]

    rows: list[dict[str, Any]] = []

    for grader_name, fn in DETERMINISTIC_GRADERS.items():
        if (ticket_id, grader_name) in completed:
            stats.skipped += 1
            continue
        error: dict[str, str] | None = None
        result: GraderResult | None = None
        try:
            result = fn(ctx)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        rows.append(
            build_grade_record(
                run_id=run_id,
                model=model,
                prompt_version=prompt_version,
                ticket_id=ticket_id,
                grader_name=grader_name,
                grader_type="deterministic",
                grader_version=DETERMINISTIC_GRADER_VERSION,
                judge_model=None,
                judge_pass_threshold=None,
                result=result,
                error=error,
            )
        )
        stats.graded += 1
        if error is not None:
            stats.errored += 1

    for grader_name, fn in LLM_JUDGE_GRADERS.items():
        if (ticket_id, grader_name) in completed:
            stats.skipped += 1
            continue
        judge_version = LLM_JUDGE_GRADER_VERSIONS[grader_name]
        error = None
        result = None
        async with semaphore:
            try:
                if cfg.dry_run:
                    call_fn = lambda: judge_stub(grader_name, ctx)  # noqa: E731
                else:
                    call_fn = lambda: fn(client, ctx, cfg.judge_model, judge_version)  # noqa: E731
                result = await call_with_retry(call_fn, cfg.max_retries)
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
        rows.append(
            build_grade_record(
                run_id=run_id,
                model=model,
                prompt_version=prompt_version,
                ticket_id=ticket_id,
                grader_name=grader_name,
                grader_type="llm_judge",
                grader_version=judge_version,
                judge_model=cfg.judge_model,
                judge_pass_threshold=JUDGE_PASS_THRESHOLD,
                result=result,
                error=error,
            )
        )
        stats.graded += 1
        if error is not None:
            stats.errored += 1

    if not rows:
        return

    async with write_lock:
        for row in rows:
            out_fh.write(json.dumps(row) + "\n")
        out_fh.flush()

    ok = sum(1 for r in rows if r["error"] is None)
    print(f"  graded {ticket_id}  {ok}/{len(rows)} grader rows written")


async def grade(cfg: GradeConfig) -> None:
    run_path = cfg.out_dir / f"{cfg.run_id}.jsonl"
    if not run_path.exists():
        raise FileNotFoundError(f"no run output found at {run_path}")

    gold_labels = load_gold_labels(cfg.gold_path)
    run_records = load_latest_run_records(run_path)

    ticket_ids = sorted(run_records)
    if cfg.limit is not None:
        ticket_ids = ticket_ids[: cfg.limit]

    missing_gold = [tid for tid in ticket_ids if tid not in gold_labels]
    if missing_gold:
        raise KeyError(f"no gold label for ticket ids: {missing_gold[:5]}...")

    out_path = cfg.out_dir / f"{cfg.run_id}.grades.jsonl"
    meta_path = cfg.out_dir / f"{cfg.run_id}.grades.meta.json"

    completed = load_completed_grade_keys(out_path)

    print(f"run_id={cfg.run_id}")
    print(f"judge_model={cfg.judge_model} dry_run={cfg.dry_run}")
    print(f"tickets to grade: {len(ticket_ids)}  already-completed grader rows: {len(completed)}")
    print(f"output: {out_path}")

    meta_path.write_text(
        json.dumps(
            {
                "run_id": cfg.run_id,
                "graded_run_output": str(run_path),
                "gold_path": str(cfg.gold_path),
                "judge_model": cfg.judge_model,
                "deterministic_grader_version": DETERMINISTIC_GRADER_VERSION,
                "llm_judge_grader_versions": LLM_JUDGE_GRADER_VERSIONS,
                "judge_pass_threshold": JUDGE_PASS_THRESHOLD,
                "limit": cfg.limit,
                "concurrency": cfg.concurrency,
                "dry_run": cfg.dry_run,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )

    client = None
    if not cfg.dry_run:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it, or pass --dry-run to "
                "smoke-test grading without calling the judge model."
            )
        client = anthropic.AsyncAnthropic(api_key=api_key)

    semaphore = asyncio.Semaphore(cfg.concurrency)
    write_lock = asyncio.Lock()
    stats = GradeStats()
    started = time.monotonic()

    try:
        with out_path.open("a") as out_fh:
            tasks = [
                grade_ticket(
                    tid,
                    run_records[tid],
                    gold_labels[tid],
                    cfg,
                    completed,
                    semaphore,
                    write_lock,
                    out_fh,
                    client,
                    stats,
                )
                for tid in ticket_ids
            ]
            await asyncio.gather(*tasks)
    finally:
        if client is not None:
            await client.close()

    elapsed = time.monotonic() - started
    print(
        f"\ndone in {elapsed:.1f}s: {stats.graded} grader rows written "
        f"({stats.errored} errored), {stats.skipped} already-completed rows skipped"
    )
    if stats.errored:
        print(f"Re-run with the same --run-id {cfg.run_id!r} to retry the errored rows.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-id", required=True, help="The run to grade (results/<run_id>.jsonl).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="Only grade the first N tickets.")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use stub judge scores (no network, no cost) instead of calling the "
        "Anthropic API. Deterministic graders always run for real either way. "
        "For verifying plumbing/resumability.",
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

    cfg = GradeConfig(
        run_id=args.run_id,
        out_dir=args.out_dir,
        gold_path=args.gold,
        judge_model=args.judge_model,
        limit=args.limit,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
    )

    asyncio.run(grade(cfg))


if __name__ == "__main__":
    main()
