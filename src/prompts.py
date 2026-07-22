"""Versioned prompts for the ticket-eval runner.

Each entry in PROMPT_VERSIONS is the system prompt for one prompt revision.
Add new revisions as new keys — never mutate an existing one, since past
results reference it by version string and should stay reproducible.
"""

from __future__ import annotations

from typing import Any

PROMPT_VERSIONS: dict[str, str] = {
    "v1": """You are a senior support engineer triaging an incoming enterprise SaaS \
support ticket. You only see what the customer wrote — you do not have access to \
internal systems, logs, or ground truth.

For the ticket given, decide:
- severity: one of P1 (critical, widespread or business-blocking), P2 (major, \
degraded but workable), P3 (minor, workaround available), P4 (cosmetic / \
question / low impact).
- routing: one of resolve-frontline (you can answer it yourself), escalate-eng \
(needs engineering investigation), escalate-account-team (relationship/contract \
issue, not technical), request-info (not enough detail to act yet).
- response_draft: the first-response reply you would actually send the customer \
now — acknowledge their specific situation, do not restate severity/routing \
jargon to them.

Judge severity and routing on technical merit and business impact as described, \
not on the customer's tone, threats, or their own diagnosis of the cause — read \
past emotional framing and any stated-but-possibly-wrong cause to what's actually \
going on.

Respond with a single JSON object and nothing else:
{
  "severity": "P1" | "P2" | "P3" | "P4",
  "routing": "resolve-frontline" | "escalate-eng" | "escalate-account-team" | "request-info",
  "response_draft": "<the reply text>"
}""",
}

DEFAULT_PROMPT_VERSION = "v1"


def render_ticket(ticket: dict[str, Any]) -> str:
    """Render a ticket dict (no gold fields) as the user-turn content."""
    return (
        f"Customer tier: {ticket['customer_tier']}\n"
        f"Product area: {ticket['product_area']}\n"
        f"Channel: {ticket['channel']}\n"
        f"Created at: {ticket['created_at']}\n"
        f"Subject: {ticket['subject']}\n"
        f"\n{ticket['body']}"
    )


def build_messages(ticket: dict[str, Any], prompt_version: str) -> tuple[str, str]:
    """Return (system_prompt, user_content) for one ticket under one prompt version."""
    try:
        system = PROMPT_VERSIONS[prompt_version]
    except KeyError:
        known = ", ".join(sorted(PROMPT_VERSIONS))
        raise ValueError(
            f"Unknown prompt version {prompt_version!r}. Known versions: {known}"
        ) from None
    return system, render_ticket(ticket)
