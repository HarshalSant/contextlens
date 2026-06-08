"""Tests for waste detection heuristics."""

from datetime import datetime

from contextlens.decompose import decompose_trace_turns
from contextlens.detectors import (
    detect_duplicates,
    detect_stale_tool_results,
    detect_unused_tool_schemas,
)
from contextlens.costs import default_cost_model
from contextlens.models import Region, Trace, TurnSnapshot, WasteKind


def _trace(turns: list[TurnSnapshot]) -> Trace:
    trace = Trace(run_id="t", model="claude-3-5-sonnet-20241022", provider="anthropic", turns=turns)
    decompose_trace_turns(trace.turns)
    return trace


def _snap(
    turn_index: int,
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_calls_made: list[str] | None = None,
    system: str = "System.",
) -> TurnSnapshot:
    return TurnSnapshot(
        turn_index=turn_index,
        timestamp=datetime.utcnow(),
        model="claude-3-5-sonnet-20241022",
        provider="anthropic",
        tool_names_defined=[t["name"] for t in (tools or [])],
        tool_names_called=tool_calls_made or [],
        raw_request={
            "system": system,
            "messages": messages,
            **({"tools": tools} if tools else {}),
        },
    )


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def test_exact_duplicate_detected() -> None:
    repeated = "This exact sentence is repeated verbatim on multiple turns."
    turns = [
        _snap(0, [{"role": "user", "content": repeated}]),
        _snap(1, [{"role": "user", "content": repeated}]),
        _snap(2, [{"role": "user", "content": repeated}]),
    ]
    trace = _trace(turns)
    findings = detect_duplicates(trace, default_cost_model)
    dup_findings = [f for f in findings if f.kind == WasteKind.DUPLICATE]
    assert len(dup_findings) > 0


def test_unique_content_not_flagged() -> None:
    turns = [
        _snap(0, [{"role": "user", "content": "Turn zero unique content here."}]),
        _snap(1, [{"role": "user", "content": "Turn one completely different content."}]),
    ]
    trace = _trace(turns)
    findings = detect_duplicates(trace, default_cost_model)
    dup_findings = [f for f in findings if f.kind == WasteKind.DUPLICATE]
    # System prompt will be duplicated but user messages won't
    user_dup = [
        f for f in dup_findings
        if "Turn zero" in f.description or "Turn one" in f.description
    ]
    assert len(user_dup) == 0


# ---------------------------------------------------------------------------
# Stale tool result detection
# ---------------------------------------------------------------------------


def test_stale_tool_result_detected() -> None:
    long_result = "tool result: " + "detailed database output row data " * 30
    turns = [
        _snap(0, [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": long_result}],
            },
            {"role": "assistant", "content": "I see, interesting."},
        ]),
        _snap(1, [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": long_result}],
            },
            {"role": "assistant", "content": "Moving on to a completely unrelated topic now."},
        ]),
        _snap(2, [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": long_result}],
            },
            {"role": "assistant", "content": "Discussing something else entirely different from before."},
        ]),
    ]
    trace = _trace(turns)
    findings = detect_stale_tool_results(trace, default_cost_model)
    assert len(findings) > 0
    assert all(f.kind == WasteKind.STALE_TOOL_RESULT for f in findings)


def test_referenced_tool_result_not_stale() -> None:
    result_text = "The user account has status: active, email: test@example.com, plan: pro"
    turns = [
        _snap(0, [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu_2", "content": result_text}],
            },
            {"role": "assistant", "content": "The account status is active with email test@example.com on plan pro."},
        ]),
    ]
    trace = _trace(turns)
    findings = detect_stale_tool_results(trace, default_cost_model)
    # The assistant references words from the result — should not be stale
    # (single turn, no future turns to check against anyway)
    assert len(findings) == 0


# ---------------------------------------------------------------------------
# Unused tool schema detection
# ---------------------------------------------------------------------------


def test_unused_tool_schema_detected() -> None:
    tool_used = {"name": "search_code", "description": "Search.", "input_schema": {"type": "object"}}
    tool_unused = {"name": "send_email", "description": "Send email.", "input_schema": {"type": "object"}}

    turns = [
        _snap(0, [{"role": "user", "content": "Do something."}],
              tools=[tool_used, tool_unused], tool_calls_made=["search_code"]),
        _snap(1, [{"role": "user", "content": "Continue."}],
              tools=[tool_used, tool_unused], tool_calls_made=["search_code"]),
        _snap(2, [{"role": "user", "content": "Finish."}],
              tools=[tool_used, tool_unused], tool_calls_made=[]),
    ]
    trace = _trace(turns)
    findings = detect_unused_tool_schemas(trace, default_cost_model)
    unused_names = [f.description for f in findings if f.kind == WasteKind.UNUSED_TOOL_SCHEMA]
    assert any("send_email" in d for d in unused_names)


def test_used_tool_not_flagged() -> None:
    tool = {"name": "run_tests", "description": "Run tests.", "input_schema": {"type": "object"}}
    turns = [
        _snap(0, [{"role": "user", "content": "Run tests please."}],
              tools=[tool], tool_calls_made=["run_tests"]),
    ]
    trace = _trace(turns)
    findings = detect_unused_tool_schemas(trace, default_cost_model)
    assert not any(
        "run_tests" in f.description
        for f in findings
        if f.kind == WasteKind.UNUSED_TOOL_SCHEMA
    )


def test_no_tools_no_findings() -> None:
    turns = [_snap(0, [{"role": "user", "content": "Hello"}])]
    trace = _trace(turns)
    findings = detect_unused_tool_schemas(trace, default_cost_model)
    assert len(findings) == 0
