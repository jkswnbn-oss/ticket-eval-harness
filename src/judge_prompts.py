"""Versioned prompts for the LLM-as-judge graders.

Mirrors src/prompts.py: each judge has its own PROMPT_VERSIONS-style dict,
keyed by version string. Add new revisions as new keys — never mutate an
existing one, since past grade records reference a judge prompt by version
and should stay reproducible. The two judges version independently since
their rubrics evolve on separate timelines.

Every judge is instructed to call the `submit_judgment` tool with a 1-5
`score` and a non-empty `reason` — forced structured output via tool use,
matching the pattern src/generate.py already uses for the dataset generator.
"""

from __future__ import annotations

from typing import Any

RESPONSE_QUALITY_JUDGE_PROMPTS: dict[str, str] = {
    "v1": """You are grading the quality of a support agent's first-response draft to an \
enterprise SaaS customer. You will see the customer's original ticket and the agent's \
draft reply. You will NOT see the agent's internal severity/routing decision — grade the \
reply as the customer would read it.

Score 1-5:
- 5: Directly addresses this customer's specific situation, correct tone, asks for or \
acknowledges concrete details from the ticket, no internal jargon (severity levels, \
routing/team names) leaked to the customer.
- 3: Generic but not wrong — reads like a template, misses some specifics from the ticket.
- 1: Ignores what the customer actually said, wrong tone (dismissive of a serious issue, \
or alarmed over a minor one), or leaks internal jargon verbatim to the customer.

Call submit_judgment with your score and a one-sentence reason citing specifics from the \
reply.""",
}

TRIAGE_REASONING_JUDGE_PROMPTS: dict[str, str] = {
    "v1": """You are grading whether a support agent's severity and routing decision on an \
enterprise SaaS ticket is defensible, given what was ACTUALLY going on — not just whether \
it matches the gold label exactly. You will see the customer's ticket, the agent's chosen \
severity and routing, and the ground truth (the real underlying issue and the facts a good \
response needed to address).

Score 1-5:
- 5: The decision reflects the real underlying issue and its actual business impact, not \
just the customer's own framing or threats. A severity/routing that differs from the exact \
gold label can still score high here if it's a reasonable call given the facts.
- 3: Partially right — got the general shape of the issue but missed a key fact that should \
have changed the severity or routing.
- 1: The decision only makes sense if you accept the customer's stated (and wrong) \
diagnosis, or ignores facts that were available and consequential (e.g. treats a \
platform-wide outage as routine, or escalates a minor cosmetic issue).

Call submit_judgment with your score and a one-sentence reason that names the specific fact \
the decision did or didn't account for.""",
}

DEFAULT_RESPONSE_QUALITY_VERSION = "v1"
DEFAULT_TRIAGE_REASONING_VERSION = "v1"

SUBMIT_JUDGMENT_TOOL: dict[str, Any] = {
    "name": "submit_judgment",
    "description": "Submit your graded score and reason for this ticket.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "1 (worst) to 5 (best).",
            },
            "reason": {
                "type": "string",
                "description": "One sentence justifying the score with specifics.",
            },
        },
        "required": ["score", "reason"],
    },
}


def render_response_quality_prompt(version: str) -> str:
    try:
        return RESPONSE_QUALITY_JUDGE_PROMPTS[version]
    except KeyError:
        known = ", ".join(sorted(RESPONSE_QUALITY_JUDGE_PROMPTS))
        raise ValueError(
            f"Unknown response_quality judge version {version!r}. Known versions: {known}"
        ) from None


def render_triage_reasoning_prompt(version: str) -> str:
    try:
        return TRIAGE_REASONING_JUDGE_PROMPTS[version]
    except KeyError:
        known = ", ".join(sorted(TRIAGE_REASONING_JUDGE_PROMPTS))
        raise ValueError(
            f"Unknown triage_reasoning judge version {version!r}. Known versions: {known}"
        ) from None
