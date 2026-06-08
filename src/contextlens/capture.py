"""Capture module: live interception and offline JSON ingestion."""

from __future__ import annotations

import json
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from .models import Trace, TurnSnapshot


class TraceCollector:
    """Accumulates TurnSnapshots; used by both live and offline paths."""

    def __init__(
        self, model: str = "unknown", provider: str = "unknown", run_id: str | None = None
    ) -> None:
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.model = model
        self.provider = provider
        self._turns: list[TurnSnapshot] = []

    def record(self, snapshot: TurnSnapshot) -> None:
        self._turns.append(snapshot)

    def build_trace(self) -> Trace:
        return Trace(
            run_id=self.run_id,
            model=self.model,
            provider=self.provider,
            turns=list(self._turns),
        )

    def save(self, path: str) -> None:
        """Persist trace as JSON for later offline analysis."""
        trace = self.build_trace()
        data = _trace_to_dict(trace)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Live capture — thin wrappers around Anthropic / OpenAI clients
# ---------------------------------------------------------------------------


class AnthropicCapture:
    """Wraps an anthropic.Anthropic client to capture every messages.create call."""

    def __init__(self, client: Any, collector: TraceCollector) -> None:
        self._client = client
        self._collector = collector
        self._original_create = client.messages.create

    def __enter__(self) -> AnthropicCapture:
        collector = self._collector
        original = self._original_create

        def patched_create(**kwargs: Any) -> Any:
            response = original(**kwargs)
            snapshot = _anthropic_request_to_snapshot(kwargs, response, len(collector._turns))
            collector.record(snapshot)
            return response

        self._client.messages.create = patched_create
        return self

    def __exit__(self, *_: Any) -> None:
        self._client.messages.create = self._original_create


class OpenAICapture:
    """Wraps an openai.OpenAI client to capture every chat.completions.create call."""

    def __init__(self, client: Any, collector: TraceCollector) -> None:
        self._client = client
        self._collector = collector
        self._original_create = client.chat.completions.create

    def __enter__(self) -> OpenAICapture:
        collector = self._collector
        original = self._original_create

        def patched_create(**kwargs: Any) -> Any:
            response = original(**kwargs)
            snapshot = _openai_request_to_snapshot(kwargs, response, len(collector._turns))
            collector.record(snapshot)
            return response

        self._client.chat.completions.create = patched_create
        return self

    def __exit__(self, *_: Any) -> None:
        self._client.chat.completions.create = self._original_create


@contextmanager
def capture_anthropic(
    client: Any, model: str | None = None
) -> Generator[TraceCollector, None, None]:
    """Context manager that intercepts all Anthropic messages.create calls.

    Example::

        import anthropic
        client = anthropic.Anthropic()
        with capture_anthropic(client, model="claude-3-5-sonnet-20241022") as collector:
            # ... your agent loop ...
        trace = collector.build_trace()
    """
    resolved_model = model or "claude-3-5-sonnet-20241022"
    collector = TraceCollector(model=resolved_model, provider="anthropic")
    cap = AnthropicCapture(client, collector)
    with cap:
        yield collector


@contextmanager
def capture_openai(client: Any, model: str | None = None) -> Generator[TraceCollector, None, None]:
    """Context manager that intercepts all OpenAI chat.completions.create calls."""
    resolved_model = model or "gpt-4o"
    collector = TraceCollector(model=resolved_model, provider="openai")
    cap = OpenAICapture(client, collector)
    with cap:
        yield collector


# ---------------------------------------------------------------------------
# Offline ingestion from saved JSON
# ---------------------------------------------------------------------------


def load_trace(path: str) -> Trace:
    """Load a trace from a JSON file saved by TraceCollector.save() or a raw dump."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return _dict_to_trace(data)


def load_trace_from_dict(data: dict[str, Any]) -> Trace:
    """Load a trace directly from a dict (e.g. from a logging pipeline)."""
    return _dict_to_trace(data)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _anthropic_request_to_snapshot(
    kwargs: dict[str, Any],
    response: Any,
    turn_index: int,
) -> TurnSnapshot:
    model = kwargs.get("model", "unknown")
    tool_names_called: list[str] = []

    if hasattr(response, "content"):
        for block in response.content:
            if hasattr(block, "type") and block.type == "tool_use":
                tool_names_called.append(getattr(block, "name", ""))

    tool_names_defined = [t.get("name", "") for t in kwargs.get("tools", [])]

    total_tokens = 0
    if hasattr(response, "usage"):
        usage = response.usage
        total_tokens = getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)

    return TurnSnapshot(
        turn_index=turn_index,
        timestamp=datetime.utcnow(),
        model=model,
        provider="anthropic",
        raw_request=dict(kwargs),
        tool_names_defined=tool_names_defined,
        tool_names_called=tool_names_called,
        total_tokens=total_tokens,
    )


def _openai_request_to_snapshot(
    kwargs: dict[str, Any],
    response: Any,
    turn_index: int,
) -> TurnSnapshot:
    model = kwargs.get("model", "unknown")
    tool_names_called: list[str] = []

    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if hasattr(tc, "function"):
                    tool_names_called.append(tc.function.name)

    tool_names_defined = [t.get("function", {}).get("name", "") for t in kwargs.get("tools", [])]

    total_tokens = 0
    if hasattr(response, "usage") and response.usage:
        total_tokens = getattr(response.usage, "total_tokens", 0)

    return TurnSnapshot(
        turn_index=turn_index,
        timestamp=datetime.utcnow(),
        model=model,
        provider="openai",
        raw_request=dict(kwargs),
        tool_names_defined=tool_names_defined,
        tool_names_called=tool_names_called,
        total_tokens=total_tokens,
    )


def _trace_to_dict(trace: Trace) -> dict[str, Any]:
    return {
        "run_id": trace.run_id,
        "model": trace.model,
        "provider": trace.provider,
        "created_at": trace.created_at.isoformat(),
        "turns": [_snapshot_to_dict(t) for t in trace.turns],
    }


def _snapshot_to_dict(snap: TurnSnapshot) -> dict[str, Any]:
    return {
        "turn_index": snap.turn_index,
        "timestamp": snap.timestamp.isoformat(),
        "model": snap.model,
        "provider": snap.provider,
        "total_tokens": snap.total_tokens,
        "tool_names_defined": snap.tool_names_defined,
        "tool_names_called": snap.tool_names_called,
        "raw_request": snap.raw_request,
    }


def _dict_to_trace(data: dict[str, Any]) -> Trace:
    turns = [_dict_to_snapshot(t) for t in data.get("turns", [])]
    return Trace(
        run_id=data.get("run_id", str(uuid.uuid4())[:8]),
        model=data.get("model", "unknown"),
        provider=data.get("provider", "unknown"),
        turns=turns,
        created_at=_parse_dt(data.get("created_at")),
    )


def _dict_to_snapshot(data: dict[str, Any]) -> TurnSnapshot:
    return TurnSnapshot(
        turn_index=data.get("turn_index", 0),
        timestamp=_parse_dt(data.get("timestamp")),
        model=data.get("model", "unknown"),
        provider=data.get("provider", "unknown"),
        total_tokens=data.get("total_tokens", 0),
        tool_names_defined=data.get("tool_names_defined", []),
        tool_names_called=data.get("tool_names_called", []),
        raw_request=data.get("raw_request", {}),
    )


def _parse_dt(value: Any) -> datetime:
    if value is None:
        return datetime.utcnow()
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return datetime.utcnow()
