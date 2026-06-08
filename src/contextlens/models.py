"""Core data models for ContextLens."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class Region(StrEnum):
    """Logical region a content block belongs to within a context window."""

    SYSTEM = "system"
    TOOL_SCHEMA = "tool_schema"
    TOOL_RESULT = "tool_result"
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    RETRIEVED_CONTENT = "retrieved_content"
    UNKNOWN = "unknown"


class WasteKind(StrEnum):
    """Category of detected waste."""

    DUPLICATE = "duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    STALE_TOOL_RESULT = "stale_tool_result"
    UNUSED_TOOL_SCHEMA = "unused_tool_schema"
    REDUNDANT_RETRIEVAL = "redundant_retrieval"


@dataclass
class ContentBlock:
    """A single identifiable piece of content within a turn's context window."""

    block_id: str
    region: Region
    content: str
    token_count: int
    turn_index: int
    # Present for tool_result blocks
    tool_call_id: str | None = None
    tool_name: str | None = None
    # Populated by rebilling analysis
    first_seen_turn: int = 0
    last_seen_turn: int = 0
    # SHA-256 hex digest of content (populated by decomposer)
    content_hash: str = ""


@dataclass
class TurnSnapshot:
    """State of the full context window as sent on a single LLM call."""

    turn_index: int
    timestamp: datetime
    model: str
    provider: str
    blocks: list[ContentBlock] = field(default_factory=list)
    total_tokens: int = 0
    # Raw payload stored for offline analysis
    raw_request: dict[str, Any] = field(default_factory=dict)
    # Tools defined in this turn's request
    tool_names_defined: list[str] = field(default_factory=list)
    # Tool names actually called in the *assistant* response for this turn
    tool_names_called: list[str] = field(default_factory=list)


@dataclass
class Trace:
    """A complete sequence of TurnSnapshots for one agent run."""

    run_id: str
    model: str
    provider: str
    turns: list[TurnSnapshot] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def total_tokens(self) -> int:
        return sum(t.total_tokens for t in self.turns)

    def total_turns(self) -> int:
        return len(self.turns)


@dataclass
class RebillingEntry:
    """Tracks the re-billing cost of a single content block across turns."""

    block_id: str
    region: Region
    content_preview: str
    token_count: int
    first_seen_turn: int
    last_seen_turn: int
    turns_present: int
    cumulative_tokens: int
    cumulative_cost_usd: float
    tool_name: str | None = None


@dataclass
class Finding:
    """A single waste finding with a concrete fix recommendation."""

    kind: WasteKind
    severity: str  # "high" | "medium" | "low"
    description: str
    fix: str
    wasted_tokens: int
    wasted_cost_usd: float
    affected_block_ids: list[str] = field(default_factory=list)
    first_seen_turn: int = 0
    last_seen_turn: int = 0


@dataclass
class RegionSummary:
    """Aggregate token + cost stats for one Region across all turns."""

    region: Region
    total_tokens: int
    total_cost_usd: float
    block_count: int
    # Fraction of total context tokens
    fraction: float = 0.0


@dataclass
class Report:
    """Top-level output of a ContextLens analysis run."""

    trace: Trace
    region_summaries: list[RegionSummary]
    rebilling_entries: list[RebillingEntry]
    findings: list[Finding]
    total_tokens_billed: int
    total_cost_usd: float
    # Tokens that could have been saved with perfect deduplication
    recoverable_tokens: int
    recoverable_cost_usd: float

    def findings_by_severity(self) -> list[Finding]:
        order = {"high": 0, "medium": 1, "low": 2}
        return sorted(self.findings, key=lambda f: (order.get(f.severity, 9), -f.wasted_tokens))
