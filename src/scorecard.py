"""Scorecard: aggregate one or more graded runs' JSONL output.

Reads results/<run_id>.grades.jsonl for each given run_id (written by
src/grade.py) and prints:
  - per-grader mean score and pass rate, across all graded tickets
  - the same breakdown split by prompt_version (denormalized onto every
    grade row by grade.py), so e.g. v1 vs v2 sit side by side

Never calls the model under test or a judge model — pure aggregation over
already-written grade records.

Usage:
    # Scorecard for a single run:
    python src/scorecard.py claude-haiku-4-5__v1__20260805T...Z

    # Compare two runs (e.g. two prompt versions) side by side:
    python src/scorecard.py <run_id_v1> <run_id_v2>

    # Spot-check the scorecard logic against only the first N tickets:
    python src/scorecard.py <run_id> --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "results"


@dataclass
class GraderStats:
    n_scored: int = 0
    n_error: int = 0
    n_passed: int = 0
    score_sum: float = 0.0

    @property
    def mean_score(self) -> float | None:
        return self.score_sum / self.n_scored if self.n_scored else None

    @property
    def pass_rate(self) -> float | None:
        return self.n_passed / self.n_scored if self.n_scored else None


def load_grade_rows(run_id: str, out_dir: Path, limit: int | None) -> list[dict[str, Any]]:
    path = out_dir / f"{run_id}.grades.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"no grade output found at {path} — run `python src/grade.py --run-id {run_id}` first"
        )
    rows: list[dict[str, Any]] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if limit is not None:
        ticket_ids = sorted({r["ticket_id"] for r in rows})[:limit]
        keep = set(ticket_ids)
        rows = [r for r in rows if r["ticket_id"] in keep]

    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, GraderStats]:
    by_grader: dict[str, GraderStats] = defaultdict(GraderStats)
    for row in rows:
        stats = by_grader[row["grader_name"]]
        if row["error"] is not None:
            stats.n_error += 1
            continue
        stats.n_scored += 1
        stats.score_sum += row["score"]
        if row["passed"]:
            stats.n_passed += 1
    return dict(by_grader)


def aggregate_by_prompt_version(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, GraderStats]]:
    by_version: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_version[row["prompt_version"]].append(row)
    return {version: aggregate(version_rows) for version, version_rows in by_version.items()}


def _fmt_pct(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "n/a"


def _fmt_score(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def render_overall_table(by_grader: dict[str, GraderStats], grader_types: dict[str, str]) -> Table:
    table = Table(title="Per-grader summary (all runs combined)")
    table.add_column("grader")
    table.add_column("type")
    table.add_column("n")
    table.add_column("errors")
    table.add_column("mean score")
    table.add_column("pass rate")
    for grader_name in sorted(by_grader):
        stats = by_grader[grader_name]
        table.add_row(
            grader_name,
            grader_types.get(grader_name, "?"),
            str(stats.n_scored),
            str(stats.n_error),
            _fmt_score(stats.mean_score),
            _fmt_pct(stats.pass_rate),
        )
    return table


def render_prompt_version_table(
    by_version: dict[str, dict[str, GraderStats]],
) -> Table:
    versions = sorted(by_version)
    graders = sorted({g for v in by_version.values() for g in v})

    table = Table(title="Breakdown by prompt version")
    table.add_column("grader")
    for version in versions:
        table.add_column(f"{version} mean")
        table.add_column(f"{version} pass%")

    for grader_name in graders:
        row = [grader_name]
        for version in versions:
            stats = by_version[version].get(grader_name)
            row.append(_fmt_score(stats.mean_score if stats else None))
            row.append(_fmt_pct(stats.pass_rate if stats else None))
        table.add_row(*row)
    return table


def render_markdown_summary(
    run_ids: list[str],
    by_grader: dict[str, GraderStats],
    grader_types: dict[str, str],
    by_version: dict[str, dict[str, GraderStats]],
) -> str:
    lines = [
        "# Scorecard",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Runs: {', '.join(run_ids)}",
        "",
        "## Per-grader summary (all runs combined)",
        "",
        "| grader | type | n | errors | mean score | pass rate |",
        "|---|---|---|---|---|---|",
    ]
    for grader_name in sorted(by_grader):
        stats = by_grader[grader_name]
        lines.append(
            f"| {grader_name} | {grader_types.get(grader_name, '?')} | {stats.n_scored} | "
            f"{stats.n_error} | {_fmt_score(stats.mean_score)} | {_fmt_pct(stats.pass_rate)} |"
        )

    versions = sorted(by_version)
    graders = sorted({g for v in by_version.values() for g in v})
    lines += ["", "## Breakdown by prompt version", ""]
    header = "| grader | " + " | ".join(f"{v} mean | {v} pass%" for v in versions) + " |"
    sep = "|---|" + "---|" * (2 * len(versions))
    lines += [header, sep]
    for grader_name in graders:
        cells = [grader_name]
        for version in versions:
            stats = by_version[version].get(grader_name)
            cells.append(_fmt_score(stats.mean_score if stats else None))
            cells.append(_fmt_pct(stats.pass_rate if stats else None))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_ids", nargs="+", help="One or more run_ids to aggregate (results/<run_id>.grades.jsonl)."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--limit", type=int, default=None, help="Only aggregate the first N tickets per run."
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Where to write the markdown summary file."
    )
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    for run_id in args.run_ids:
        rows = load_grade_rows(run_id, args.out_dir, args.limit)
        all_rows.extend(rows)

    if not all_rows:
        print("no grade rows found for the given run_id(s)", file=sys.stderr)
        sys.exit(1)

    grader_types = {r["grader_name"]: r["grader_type"] for r in all_rows}
    by_grader = aggregate(all_rows)
    by_version = aggregate_by_prompt_version(all_rows)

    console = Console()
    console.print(render_overall_table(by_grader, grader_types))
    console.print(render_prompt_version_table(by_version))

    summary_md = render_markdown_summary(args.run_ids, by_grader, grader_types, by_version)
    out_path = args.out or (args.out_dir / f"scorecard__{'_'.join(args.run_ids)}.md")
    out_path.write_text(summary_md)
    print(f"\nwritten summary: {out_path}")


if __name__ == "__main__":
    main()
