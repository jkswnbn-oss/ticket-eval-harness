"""Generate the synthetic support ticket dataset.

Calls the Anthropic API in batches, forcing structured output via tool use,
and steers each batch toward whichever severity / reporter-profile / product-area
buckets are under-represented so far relative to the target distribution in
the project scope doc (§3):

    severity:  15% P1, 25% P2, 40% P3, 20% P4
    trap tickets (reporter_profile == "wrong-diagnosis", or a severity that's
    buried under a mundane-sounding vague-frustrated / multi-issue report):
    at least 20% of the dataset

Usage:
    ANTHROPIC_API_KEY=sk-... python src/generate.py --n-tickets 150

Writes data/tickets.json (no gold) and data/gold_labels.json (id -> gold),
validating every record against schema.py before it's written.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import GoldLabel, Ticket  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data"

PRODUCT_AREAS = [
    "search",
    "analytics",
    "integrations",
    "auth",
    "content-management",
    "ai-assistant",
]
CUSTOMER_TIERS = ["platinum", "gold", "standard"]
CHANNELS = ["email", "portal", "chat"]
REPORTER_PROFILES = [
    "precise-technical",
    "vague-frustrated",
    "multi-issue",
    "escalation-threat",
    "wrong-diagnosis",
]
SEVERITIES = ["P1", "P2", "P3", "P4"]
ROUTINGS = [
    "resolve-frontline",
    "escalate-eng",
    "escalate-account-team",
    "request-info",
]

SEVERITY_TARGET_FRACTIONS = {"P1": 0.15, "P2": 0.25, "P3": 0.40, "P4": 0.20}

TOOL_SCHEMA = {
    "name": "submit_tickets",
    "description": "Submit a batch of synthetic enterprise support tickets with their gold labels.",
    "input_schema": {
        "type": "object",
        "properties": {
            "tickets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "created_at": {
                            "type": "string",
                            "description": "ISO-8601 timestamp, e.g. 2026-06-14T09:22:00Z",
                        },
                        "customer_tier": {"type": "string", "enum": CUSTOMER_TIERS},
                        "product_area": {"type": "string", "enum": PRODUCT_AREAS},
                        "channel": {"type": "string", "enum": CHANNELS},
                        "subject": {"type": "string"},
                        "body": {
                            "type": "string",
                            "description": "The realistic, messy ticket body in the reporter's voice.",
                        },
                        "reporter_profile": {
                            "type": "string",
                            "enum": REPORTER_PROFILES,
                        },
                        "gold": {
                            "type": "object",
                            "properties": {
                                "severity": {"type": "string", "enum": SEVERITIES},
                                "true_issue": {
                                    "type": "string",
                                    "description": "One sentence: what is actually wrong, ground truth.",
                                },
                                "correct_routing": {
                                    "type": "string",
                                    "enum": ROUTINGS,
                                },
                                "key_facts_needed": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Facts a good response must acknowledge or request.",
                                },
                            },
                            "required": [
                                "severity",
                                "true_issue",
                                "correct_routing",
                                "key_facts_needed",
                            ],
                        },
                    },
                    "required": [
                        "created_at",
                        "customer_tier",
                        "product_area",
                        "channel",
                        "subject",
                        "body",
                        "reporter_profile",
                        "gold",
                    ],
                },
            },
        },
        "required": ["tickets"],
    },
}

SYSTEM_PROMPT = """You are generating a synthetic dataset of enterprise B2B SaaS support \
tickets for an eval harness. This data trains no one and represents no real company — \
invent generic SaaS product areas, generic company names if needed, and realistic but \
entirely fictional technical details (error codes, integration names, field names).

Absolute rules:
- No real company, product, or person names. No real API/vendor names beyond generic \
  category references (e.g. "our SSO provider" not "Okta", unless inventing a fictional \
  vendor name is more natural).
- Every ticket needs a `gold` block that is the ground truth a human support lead would \
  assign — NOT necessarily what the customer thinks is wrong.

Reporter profiles (vary the writing voice accordingly):
- precise-technical: clear repro steps, exact error text, already ruled things out.
- vague-frustrated: emotional, imprecise, you have to infer the actual technical issue.
- multi-issue: bundles 2-3 unrelated problems in one ticket, only one may be the real \
  urgent one.
- escalation-threat: name-drops contract renewal / churn / "escalating to my CSM" — \
  tests whether severity is assigned on technical merit, not the threat.
- wrong-diagnosis: the customer confidently states an incorrect cause (blames the wrong \
  feature, assumes a bug that's actually user error or vice versa). gold.true_issue must \
  diverge from what the customer claims.

Trap tickets matter most: a P1 buried inside a rambling vague-frustrated or multi-issue \
ticket that reads like a P3 on the surface, or a wrong-diagnosis ticket where the stated \
cause and the real cause point to different teams for routing. Roughly a fifth of the \
dataset should be traps like this.

Call the submit_tickets tool with the batch. Do not include any text outside the tool call."""


def build_batch_prompt(
    batch_size: int, counts: dict[str, Counter[str]], target_total: int
) -> str:
    def deficit_ranked(dimension: str, universe: list[str]) -> list[str]:
        c = counts[dimension]
        return sorted(universe, key=lambda k: c.get(k, 0))

    severity_need = []
    for sev, frac in SEVERITY_TARGET_FRACTIONS.items():
        target = frac * target_total
        deficit = target - counts["severity"].get(sev, 0)
        severity_need.append((deficit, sev))
    severity_need.sort(reverse=True)
    priority_severities = [s for _, s in severity_need[:2]]

    under_profiles = deficit_ranked("reporter_profile", REPORTER_PROFILES)[:2]
    under_areas = deficit_ranked("product_area", PRODUCT_AREAS)[:3]

    return (
        f"Generate exactly {batch_size} new tickets, distinct scenarios from anything "
        f"generated so far in this run.\n\n"
        f"Lean toward severities {priority_severities} and reporter profiles "
        f"{under_profiles} in this batch to help the overall dataset hit its target "
        f"distribution, but don't force every ticket into those buckets if it'd feel "
        f"contrived — realism first.\n"
        f"Favor product areas {under_areas} if natural.\n"
        f"IDs will be assigned by the caller — do not include an `id` field."
    )


def call_with_retry(client: Any, **kwargs: Any) -> Any:
    delay = 2.0
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            return client.messages.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - broad: covers rate limit / transient API errors
            last_err = e
            if attempt == 4:
                break
            print(f"  API error ({e}); retrying in {delay:.0f}s...", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Anthropic API call failed after retries: {last_err}")


def generate_dataset(
    n_tickets: int, batch_size: int, model: str, seed: int | None
) -> tuple[list[Ticket], dict[str, GoldLabel]]:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before running generate.py."
        )
    client = anthropic.Anthropic(api_key=api_key)

    if seed is not None:
        random.seed(seed)

    tickets: list[Ticket] = []
    golds: dict[str, GoldLabel] = {}
    counts: dict[str, Counter[str]] = {
        "severity": Counter(),
        "reporter_profile": Counter(),
        "product_area": Counter(),
    }
    next_id = 1

    while len(tickets) < n_tickets:
        remaining = n_tickets - len(tickets)
        this_batch = min(batch_size, remaining)
        prompt = build_batch_prompt(this_batch, counts, n_tickets)
        print(f"Requesting batch of {this_batch} ({len(tickets)}/{n_tickets} so far)...")

        response = call_with_retry(
            client,
            model=model,
            max_tokens=8000,
            temperature=1.0,
            system=SYSTEM_PROMPT,
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "submit_tickets"},
            messages=[{"role": "user", "content": prompt}],
        )

        tool_use = next(
            (b for b in response.content if getattr(b, "type", None) == "tool_use"), None
        )
        if tool_use is None:
            print("  Warning: no tool_use block in response, skipping batch", file=sys.stderr)
            continue

        raw_tickets = tool_use.input.get("tickets", [])
        for raw in raw_tickets:
            if len(tickets) >= n_tickets:
                break
            gold_raw = raw.pop("gold", None)
            ticket_id = f"TKT-{next_id:04d}"
            try:
                ticket = Ticket.model_validate({**raw, "id": ticket_id})
                gold = GoldLabel.model_validate(gold_raw)
            except ValidationError as e:
                print(f"  Dropping invalid record: {e}", file=sys.stderr)
                continue

            tickets.append(ticket)
            golds[ticket_id] = gold
            counts["severity"][gold.severity] += 1
            counts["reporter_profile"][ticket.reporter_profile] += 1
            counts["product_area"][ticket.product_area] += 1
            next_id += 1

    return tickets, golds


def write_dataset(
    tickets: list[Ticket], golds: dict[str, GoldLabel], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tickets_path = out_dir / "tickets.json"
    gold_path = out_dir / "gold_labels.json"

    tickets_path.write_text(
        json.dumps(
            [json.loads(t.model_dump_json()) for t in tickets], indent=2, sort_keys=False
        )
        + "\n"
    )
    gold_path.write_text(
        json.dumps(
            {tid: json.loads(g.model_dump_json()) for tid, g in golds.items()},
            indent=2,
            sort_keys=False,
        )
        + "\n"
    )
    print(f"Wrote {len(tickets)} tickets to {tickets_path}")
    print(f"Wrote {len(golds)} gold labels to {gold_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tickets", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5-20250929",
        help="Generation benefits from a stronger model even though eval runs use a cheaper one.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    tickets, golds = generate_dataset(
        n_tickets=args.n_tickets,
        batch_size=args.batch_size,
        model=args.model,
        seed=args.seed,
    )
    write_dataset(tickets, golds, args.out_dir)

    sev_counts = Counter(g.severity for g in golds.values())
    trap_count = sum(1 for t in tickets if t.reporter_profile == "wrong-diagnosis")
    print(f"\nSeverity distribution: {dict(sev_counts)}")
    print(f"wrong-diagnosis (trap) tickets: {trap_count} ({trap_count / len(tickets):.0%})")


if __name__ == "__main__":
    main()
