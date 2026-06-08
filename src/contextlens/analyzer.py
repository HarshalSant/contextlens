"""Orchestrates decomposition, rebilling, and waste detection into a Report."""

from __future__ import annotations

from collections import defaultdict

from .capture import load_trace
from .costs import CostModel, default_cost_model
from .decompose import decompose_trace_turns
from .detectors import run_all_detectors
from .models import Region, RegionSummary, Report, Trace
from .rebilling import compute_rebilling, recoverable_tokens


def analyze_trace(
    trace: Trace,
    cost_model: CostModel | None = None,
) -> Report:
    """Run the full ContextLens analysis pipeline on a Trace."""
    cm = cost_model or default_cost_model
    model = trace.model

    # 1. Decompose raw requests into typed blocks
    decompose_trace_turns(trace.turns)

    # 2. Re-billing analysis
    rebilling_entries = compute_rebilling(trace, cm)

    # 3. Waste detection
    findings = run_all_detectors(trace, cm)

    # 4. Region summaries
    region_token_totals: dict[Region, int] = defaultdict(int)
    region_block_counts: dict[Region, int] = defaultdict(int)
    for turn in trace.turns:
        for block in turn.blocks:
            region_token_totals[block.region] += block.token_count
            region_block_counts[block.region] += 1

    total_tokens = sum(region_token_totals.values())
    total_cost = cm.input_cost(model, total_tokens)

    region_summaries = []
    for region in Region:
        tokens = region_token_totals.get(region, 0)
        if tokens == 0:
            continue
        cost = cm.input_cost(model, tokens)
        fraction = tokens / total_tokens if total_tokens else 0.0
        region_summaries.append(
            RegionSummary(
                region=region,
                total_tokens=tokens,
                total_cost_usd=cost,
                block_count=region_block_counts.get(region, 0),
                fraction=fraction,
            )
        )
    region_summaries.sort(key=lambda s: s.total_tokens, reverse=True)

    # 5. Recoverable waste
    rec_tokens = recoverable_tokens(rebilling_entries)
    rec_cost = cm.input_cost(model, rec_tokens)

    return Report(
        trace=trace,
        region_summaries=region_summaries,
        rebilling_entries=rebilling_entries,
        findings=findings,
        total_tokens_billed=total_tokens,
        total_cost_usd=total_cost,
        recoverable_tokens=rec_tokens,
        recoverable_cost_usd=rec_cost,
    )


def analyze_file(
    path: str,
    cost_model: CostModel | None = None,
) -> Report:
    """Convenience: load a trace JSON file and run the full analysis."""
    trace = load_trace(path)
    return analyze_trace(trace, cost_model)
