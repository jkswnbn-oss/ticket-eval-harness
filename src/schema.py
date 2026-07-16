"""Pydantic schema for support tickets and gold labels.

Kept separate from generate.py/run.py so both the generation pipeline
and the grading pipeline import the same source of truth.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, RootModel

CustomerTier = Literal["platinum", "gold", "standard"]
ProductArea = Literal[
    "search",
    "analytics",
    "integrations",
    "auth",
    "content-management",
    "ai-assistant",
]
Channel = Literal["email", "portal", "chat"]
ReporterProfile = Literal[
    "precise-technical",
    "vague-frustrated",
    "multi-issue",
    "escalation-threat",
    "wrong-diagnosis",
]
Severity = Literal["P1", "P2", "P3", "P4"]
Routing = Literal[
    "resolve-frontline",
    "escalate-eng",
    "escalate-account-team",
    "request-info",
]


class Ticket(BaseModel):
    """A synthetic ticket as the model under test sees it — no gold data."""

    id: str = Field(pattern=r"^TKT-\d{4}$")
    created_at: datetime
    customer_tier: CustomerTier
    product_area: ProductArea
    channel: Channel
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)
    reporter_profile: ReporterProfile


class GoldLabel(BaseModel):
    """Ground truth for one ticket, kept out of the model-under-test's context."""

    severity: Severity
    true_issue: str = Field(min_length=1)
    correct_routing: Routing
    key_facts_needed: list[str] = Field(min_length=1)


class TicketDataset(RootModel[list[Ticket]]):
    pass


class GoldLabelSet(RootModel[dict[str, GoldLabel]]):
    pass
