"""Integration tests for the analyzer pipeline."""

from datetime import datetime

from contextlens.analyzer import analyze_trace
from contextlens.models import Region, Trace, TurnSnapshot


def _simple_trace() -> Trace:
    turns = [
        TurnSnapshot(
            turn_index=i,
            timestamp=datetime.utcnow(),
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            tool_names_defined=["search_code"],
            tool_names_called=[] if i < 3 else ["search_code"],
            raw_request={
                "system": "You are helpful.",
                "tools": [{"name": "search_code", "description": "Search.", "input_schema": {}}],
                "messages": [
                    {"role": "user", "content": f"User message turn {i}"},
                    {"role": "assistant", "content": f"Assistant response for turn {i}"},
                ],
            },
        )
        for i in range(5)
    ]
    return Trace(
        run_id="int-test", model="claude-3-5-sonnet-20241022", provider="anthropic", turns=turns
    )


def test_report_has_region_summaries() -> None:
    report = analyze_trace(_simple_trace())
    assert len(report.region_summaries) > 0


def test_total_tokens_positive() -> None:
    report = analyze_trace(_simple_trace())
    assert report.total_tokens_billed > 0


def test_total_cost_positive() -> None:
    report = analyze_trace(_simple_trace())
    assert report.total_cost_usd > 0


def test_recoverable_tokens_non_negative() -> None:
    report = analyze_trace(_simple_trace())
    assert report.recoverable_tokens >= 0


def test_findings_list() -> None:
    report = analyze_trace(_simple_trace())
    # Should always be a list (possibly empty)
    assert isinstance(report.findings, list)


def test_system_region_present() -> None:
    report = analyze_trace(_simple_trace())
    regions = {rs.region for rs in report.region_summaries}
    assert Region.SYSTEM in regions


def test_findings_sorted_by_severity() -> None:
    report = analyze_trace(_simple_trace())
    findings = report.findings_by_severity()
    order = {"high": 0, "medium": 1, "low": 2}
    severities = [order[f.severity] for f in findings]
    assert severities == sorted(severities)


def test_rebilling_entries_present() -> None:
    report = analyze_trace(_simple_trace())
    # System prompt is repeated — at minimum one rebilling entry
    assert len(report.rebilling_entries) > 0
