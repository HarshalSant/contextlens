"""Tests for the re-billing analysis module."""

from datetime import datetime

from contextlens.decompose import decompose_trace_turns
from contextlens.models import Region, Trace, TurnSnapshot
from contextlens.rebilling import compute_rebilling, recoverable_tokens, total_billed_tokens


def _make_trace_with_repeated_block() -> Trace:
    """A 3-turn trace where the same system prompt appears every turn."""
    system_content = "You are a helpful assistant with many instructions."
    turns = []
    for i in range(3):
        snap = TurnSnapshot(
            turn_index=i,
            timestamp=datetime.utcnow(),
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            raw_request={
                "system": system_content,
                "messages": [{"role": "user", "content": f"Turn {i} user message"}],
            },
        )
        turns.append(snap)
    trace = Trace(run_id="test-001", model="claude-3-5-sonnet-20241022", provider="anthropic", turns=turns)
    decompose_trace_turns(trace.turns)
    return trace


def test_rebilling_entries_exist() -> None:
    trace = _make_trace_with_repeated_block()
    entries = compute_rebilling(trace)
    assert len(entries) > 0


def test_system_prompt_rebilled_3_turns() -> None:
    trace = _make_trace_with_repeated_block()
    entries = compute_rebilling(trace)
    system_entries = [e for e in entries if e.region == Region.SYSTEM]
    assert len(system_entries) > 0
    entry = system_entries[0]
    # System prompt appears on all 3 turns
    assert entry.turns_present == 3
    assert entry.cumulative_tokens == entry.token_count * 3


def test_cumulative_cost_positive() -> None:
    trace = _make_trace_with_repeated_block()
    entries = compute_rebilling(trace)
    system_entries = [e for e in entries if e.region == Region.SYSTEM]
    assert system_entries[0].cumulative_cost_usd > 0


def test_recoverable_tokens_less_than_total() -> None:
    trace = _make_trace_with_repeated_block()
    entries = compute_rebilling(trace)
    rec = recoverable_tokens(entries)
    total = total_billed_tokens(trace)
    assert 0 <= rec < total


def test_recoverable_tokens_math() -> None:
    """Recoverable = token_count * (turns_present - 1) summed over blocks seen > 1 turn."""
    trace = _make_trace_with_repeated_block()
    entries = compute_rebilling(trace)
    expected = sum(e.token_count * (e.turns_present - 1) for e in entries if e.turns_present > 1)
    assert recoverable_tokens(entries) == expected


def test_unique_block_not_rebilled() -> None:
    """A block that appears only once should have turns_present == 1."""
    turns = [
        TurnSnapshot(
            turn_index=0,
            timestamp=datetime.utcnow(),
            model="gpt-4o",
            provider="openai",
            raw_request={"messages": [{"role": "user", "content": "Only once."}]},
        )
    ]
    trace = Trace(run_id="t", model="gpt-4o", provider="openai", turns=turns)
    decompose_trace_turns(trace.turns)
    entries = compute_rebilling(trace)
    user_entries = [e for e in entries if e.region == Region.USER_MESSAGE]
    assert all(e.turns_present == 1 for e in user_entries)


def test_entries_sorted_by_cost_descending() -> None:
    trace = _make_trace_with_repeated_block()
    entries = compute_rebilling(trace)
    costs = [e.cumulative_cost_usd for e in entries]
    assert costs == sorted(costs, reverse=True)
