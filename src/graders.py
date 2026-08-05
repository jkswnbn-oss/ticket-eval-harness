"""Grader functions for M3.

Two families, kept in clearly separate sections of this module and never
mixed in one function:

- Deterministic graders: pure, synchronous, no API calls. Rule-based checks
  on the structured fields of a parsed model output.
- LLM-as-judge graders: async, call the Anthropic API, for the subjective
  parts (response quality, whether the triage reasoning holds up).

Every grader returns a GraderResult(score, passed, reason) — score is always
normalized to 0.0-1.0 regardless of grader type, so the scorecard can average
across deterministic and judge graders uniformly. `reason` is required and
non-empty; no grader ever returns a bare number.

Grading reads an already-written run record (see src/run.py's per-ticket
record shape) plus its matching src/schema.py:GoldLabel — it never calls the
model under test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, get_args

from judge_prompts import (
    SUBMIT_JUDGMENT_TOOL,
    render_response_quality_prompt,
    render_triage_reasoning_prompt,
)
from schema import GoldLabel, Routing, Severity

VALID_SEVERITIES = set(get_args(Severity))
VALID_ROUTINGS = set(get_args(Routing))

DETERMINISTIC_GRADER_VERSION = "det-v1"


@dataclass
class GraderResult:
    score: float  # normalized 0.0-1.0
    passed: bool
    reason: str


@dataclass
class GradingContext:
    """Everything one grader call needs for one (run, ticket) pair."""

    ticket_id: str
    raw_output: str | None
    ticket_input: dict[str, Any]  # the run record's `input` field
    gold: GoldLabel

    @property
    def parsed(self) -> dict[str, Any] | None:
        if not self.raw_output:
            return None
        try:
            value = json.loads(self.raw_output)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None


# --- deterministic graders -------------------------------------------------


def grade_output_parses(ctx: GradingContext) -> GraderResult:
    if ctx.parsed is not None:
        return GraderResult(score=1.0, passed=True, reason="raw_output parses as a JSON object")
    return GraderResult(
        score=0.0, passed=False, reason="raw_output is empty or not valid JSON"
    )


def grade_has_required_fields(ctx: GradingContext) -> GraderResult:
    parsed = ctx.parsed
    if parsed is None:
        return GraderResult(score=0.0, passed=False, reason="output did not parse")
    required = ("severity", "routing", "response_draft")
    missing = [f for f in required if f not in parsed]
    if not missing:
        return GraderResult(score=1.0, passed=True, reason="all required fields present")
    return GraderResult(
        score=0.0, passed=False, reason=f"missing fields: {', '.join(missing)}"
    )


def grade_valid_severity(ctx: GradingContext) -> GraderResult:
    parsed = ctx.parsed
    if parsed is None or "severity" not in parsed:
        return GraderResult(score=0.0, passed=False, reason="no severity field to validate")
    value = parsed["severity"]
    if value in VALID_SEVERITIES:
        return GraderResult(score=1.0, passed=True, reason=f"severity {value!r} is a valid enum value")
    return GraderResult(
        score=0.0, passed=False, reason=f"severity {value!r} is not one of {sorted(VALID_SEVERITIES)}"
    )


def grade_valid_routing(ctx: GradingContext) -> GraderResult:
    parsed = ctx.parsed
    if parsed is None or "routing" not in parsed:
        return GraderResult(score=0.0, passed=False, reason="no routing field to validate")
    value = parsed["routing"]
    if value in VALID_ROUTINGS:
        return GraderResult(score=1.0, passed=True, reason=f"routing {value!r} is a valid enum value")
    return GraderResult(
        score=0.0, passed=False, reason=f"routing {value!r} is not one of {sorted(VALID_ROUTINGS)}"
    )


def grade_severity_exact_match(ctx: GradingContext) -> GraderResult:
    parsed = ctx.parsed
    if parsed is None or "severity" not in parsed:
        return GraderResult(score=0.0, passed=False, reason="no severity field to compare")
    value = parsed["severity"]
    gold = ctx.gold.severity
    if value == gold:
        return GraderResult(score=1.0, passed=True, reason=f"severity {value!r} matches gold {gold!r}")
    return GraderResult(
        score=0.0, passed=False, reason=f"severity {value!r} does not match gold {gold!r}"
    )


def grade_routing_exact_match(ctx: GradingContext) -> GraderResult:
    parsed = ctx.parsed
    if parsed is None or "routing" not in parsed:
        return GraderResult(score=0.0, passed=False, reason="no routing field to compare")
    value = parsed["routing"]
    gold = ctx.gold.correct_routing
    if value == gold:
        return GraderResult(score=1.0, passed=True, reason=f"routing {value!r} matches gold {gold!r}")
    return GraderResult(
        score=0.0, passed=False, reason=f"routing {value!r} does not match gold {gold!r}"
    )


DETERMINISTIC_GRADERS: dict[str, Callable[[GradingContext], GraderResult]] = {
    "output_parses": grade_output_parses,
    "has_required_fields": grade_has_required_fields,
    "valid_severity": grade_valid_severity,
    "valid_routing": grade_valid_routing,
    "severity_exact_match": grade_severity_exact_match,
    "routing_exact_match": grade_routing_exact_match,
}


# --- llm-as-judge graders ----------------------------------------------------

JUDGE_PASS_THRESHOLD = 0.6  # 3/5


async def _call_judge(
    client: Any, model: str, system: str, user_content: str, max_tokens: int = 512
) -> GraderResult:
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_content}],
        tools=[SUBMIT_JUDGMENT_TOOL],
        tool_choice={"type": "tool", "name": "submit_judgment"},
    )
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_judgment":
            raw_score = int(block.input["score"])
            reason = str(block.input["reason"])
            score = raw_score / 5.0
            return GraderResult(score=score, passed=score >= JUDGE_PASS_THRESHOLD, reason=reason)
    raise RuntimeError("judge model did not call submit_judgment")


def _render_ticket_block(ticket_input: dict[str, Any]) -> str:
    return (
        f"Customer tier: {ticket_input['customer_tier']}\n"
        f"Product area: {ticket_input['product_area']}\n"
        f"Channel: {ticket_input['channel']}\n"
        f"Subject: {ticket_input['subject']}\n"
        f"\n{ticket_input['body']}"
    )


async def grade_response_quality(
    client: Any, ctx: GradingContext, judge_model: str, judge_version: str
) -> GraderResult:
    parsed = ctx.parsed
    if parsed is None or "response_draft" not in parsed:
        return GraderResult(
            score=0.0, passed=False, reason="no response_draft to judge (output did not parse)"
        )
    system = render_response_quality_prompt(judge_version)
    user_content = (
        f"=== Customer ticket ===\n{_render_ticket_block(ctx.ticket_input)}\n\n"
        f"=== Agent's draft reply ===\n{parsed['response_draft']}"
    )
    return await _call_judge(client, judge_model, system, user_content)


async def grade_triage_reasoning(
    client: Any, ctx: GradingContext, judge_model: str, judge_version: str
) -> GraderResult:
    parsed = ctx.parsed
    if parsed is None or "severity" not in parsed or "routing" not in parsed:
        return GraderResult(
            score=0.0,
            passed=False,
            reason="no severity/routing to judge (output did not parse)",
        )
    system = render_triage_reasoning_prompt(judge_version)
    key_facts = "\n".join(f"- {fact}" for fact in ctx.gold.key_facts_needed)
    user_content = (
        f"=== Customer ticket ===\n{_render_ticket_block(ctx.ticket_input)}\n\n"
        f"=== Agent's decision ===\n"
        f"severity: {parsed['severity']}\n"
        f"routing: {parsed['routing']}\n\n"
        f"=== Ground truth (not seen by the agent) ===\n"
        f"Real issue: {ctx.gold.true_issue}\n"
        f"Facts a good response needed to address:\n{key_facts}"
    )
    return await _call_judge(client, judge_model, system, user_content)


LLM_JUDGE_GRADERS: dict[
    str, Callable[[Any, GradingContext, str, str], Any]
] = {
    "response_quality": grade_response_quality,
    "triage_reasoning": grade_triage_reasoning,
}

LLM_JUDGE_GRADER_VERSIONS: dict[str, str] = {
    "response_quality": "v1",
    "triage_reasoning": "v1",
}
