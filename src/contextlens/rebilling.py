"""Re-billing analysis: track how much each content block costs across turns."""

from __future__ import annotations

from collections import defaultdict

from .costs import CostModel, default_cost_model
from .models import ContentBlock, RebillingEntry, Region, Trace


def compute_rebilling(
    trace: Trace,
    cost_model: CostModel | None = None,
) -> list[RebillingEntry]:
    """Compute per-block cumulative re-billing across all turns in the trace.

    A block is considered re-billed on every turn after its first appearance
    where an identical (by content_hash) block is present.  We match blocks
    across turns using content_hash, so identity is content-based, not
    positional.
    """
    cm = cost_model or default_cost_model
    model = trace.model

    # Group blocks by content_hash across all turns
    # hash -> list of (turn_index, block)
    hash_to_appearances: dict[str, list[tuple[int, ContentBlock]]] = defaultdict(list)

    for turn in trace.turns:
        for block in turn.blocks:
            if block.content_hash and block.region not in (Region.UNKNOWN,):
                hash_to_appearances[block.content_hash].append((turn.turn_index, block))

    entries: list[RebillingEntry] = []

    for _content_hash, appearances in hash_to_appearances.items():
        if not appearances:
            continue

        appearances_sorted = sorted(appearances, key=lambda x: x[0])
        first_turn = appearances_sorted[0][0]
        last_turn = appearances_sorted[-1][0]
        representative_block = appearances_sorted[0][1]

        turns_present = len(appearances_sorted)
        cumulative_tokens = representative_block.token_count * turns_present
        cumulative_cost = cm.input_cost(model, cumulative_tokens)

        # Update block metadata
        for _, block in appearances:
            block.first_seen_turn = first_turn
            block.last_seen_turn = last_turn

        content_preview = representative_block.content[:120].replace("\n", " ")
        if len(representative_block.content) > 120:
            content_preview += "…"

        entries.append(
            RebillingEntry(
                block_id=representative_block.block_id,
                region=representative_block.region,
                content_preview=content_preview,
                token_count=representative_block.token_count,
                first_seen_turn=first_turn,
                last_seen_turn=last_turn,
                turns_present=turns_present,
                cumulative_tokens=cumulative_tokens,
                cumulative_cost_usd=cumulative_cost,
                tool_name=representative_block.tool_name,
            )
        )

    # Sort by cumulative cost descending — most expensive at top
    entries.sort(key=lambda e: e.cumulative_cost_usd, reverse=True)
    return entries


def total_billed_tokens(trace: Trace) -> int:
    """Sum of all tokens across all turns — what you actually paid for."""
    return sum(
        block.token_count
        for turn in trace.turns
        for block in turn.blocks
    )


def recoverable_tokens(rebilling_entries: list[RebillingEntry]) -> int:
    """Tokens that would have been saved if every block was sent only once."""
    saved = 0
    for entry in rebilling_entries:
        if entry.turns_present > 1:
            # The first send is unavoidable; every subsequent re-send is waste
            saved += entry.token_count * (entry.turns_present - 1)
    return saved
