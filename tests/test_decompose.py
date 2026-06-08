"""Tests for the decompose module."""

from contextlens.decompose import decompose_snapshot
from contextlens.models import Region, TurnSnapshot
from datetime import datetime


def _make_snap(raw_request: dict, provider: str = "anthropic") -> TurnSnapshot:
    return TurnSnapshot(
        turn_index=0,
        timestamp=datetime.utcnow(),
        model="claude-3-5-sonnet-20241022",
        provider=provider,
        raw_request=raw_request,
    )


# ---------------------------------------------------------------------------
# Anthropic decomposition
# ---------------------------------------------------------------------------


def test_system_prompt_extracted() -> None:
    snap = _make_snap({"system": "You are helpful.", "messages": []})
    blocks = decompose_snapshot(snap)
    system_blocks = [b for b in blocks if b.region == Region.SYSTEM]
    assert len(system_blocks) == 1
    assert "helpful" in system_blocks[0].content


def test_tool_schema_extracted() -> None:
    tool = {"name": "search_code", "description": "Search.", "input_schema": {"type": "object"}}
    snap = _make_snap({"tools": [tool], "messages": []})
    blocks = decompose_snapshot(snap)
    schema_blocks = [b for b in blocks if b.region == Region.TOOL_SCHEMA]
    assert len(schema_blocks) == 1
    assert schema_blocks[0].tool_name == "search_code"


def test_user_message_extracted() -> None:
    snap = _make_snap({
        "messages": [{"role": "user", "content": "Hello, world!"}]
    })
    blocks = decompose_snapshot(snap)
    user_blocks = [b for b in blocks if b.region == Region.USER_MESSAGE]
    assert len(user_blocks) == 1
    assert "Hello" in user_blocks[0].content


def test_assistant_message_extracted() -> None:
    snap = _make_snap({
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello back!"},
        ]
    })
    blocks = decompose_snapshot(snap)
    asst_blocks = [b for b in blocks if b.region == Region.ASSISTANT_MESSAGE]
    assert len(asst_blocks) == 1


def test_tool_result_extracted() -> None:
    snap = _make_snap({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "Result data here."}
                ],
            }
        ]
    })
    blocks = decompose_snapshot(snap)
    tr_blocks = [b for b in blocks if b.region == Region.TOOL_RESULT]
    assert len(tr_blocks) == 1
    assert tr_blocks[0].tool_call_id == "tu_1"


def test_tool_use_block_in_assistant() -> None:
    snap = _make_snap({
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_2", "name": "search_code", "input": {"pattern": "foo"}},
                ],
            }
        ]
    })
    blocks = decompose_snapshot(snap)
    asst_blocks = [b for b in blocks if b.region == Region.ASSISTANT_MESSAGE]
    assert len(asst_blocks) == 1
    assert asst_blocks[0].tool_name == "search_code"
    assert asst_blocks[0].tool_call_id == "tu_2"


def test_content_hash_populated() -> None:
    snap = _make_snap({"messages": [{"role": "user", "content": "Test content"}]})
    blocks = decompose_snapshot(snap)
    for b in blocks:
        assert len(b.content_hash) == 64  # SHA-256 hex digest


def test_token_count_positive() -> None:
    snap = _make_snap({"messages": [{"role": "user", "content": "Some text with a reasonable length"}]})
    blocks = decompose_snapshot(snap)
    for b in blocks:
        assert b.token_count > 0


# ---------------------------------------------------------------------------
# OpenAI decomposition
# ---------------------------------------------------------------------------


def test_openai_system_message() -> None:
    snap = _make_snap(
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ]
        },
        provider="openai",
    )
    blocks = decompose_snapshot(snap)
    system_blocks = [b for b in blocks if b.region == Region.SYSTEM]
    assert len(system_blocks) == 1


def test_openai_tool_result_message() -> None:
    snap = _make_snap(
        {
            "messages": [
                {"role": "tool", "tool_call_id": "call_1", "content": "Tool output here."},
            ]
        },
        provider="openai",
    )
    blocks = decompose_snapshot(snap)
    tr_blocks = [b for b in blocks if b.region == Region.TOOL_RESULT]
    assert len(tr_blocks) == 1
    assert tr_blocks[0].tool_call_id == "call_1"


def test_openai_tool_schema() -> None:
    snap = _make_snap(
        {
            "tools": [
                {"type": "function", "function": {"name": "my_tool", "description": "Does something"}}
            ],
            "messages": [],
        },
        provider="openai",
    )
    blocks = decompose_snapshot(snap)
    schema_blocks = [b for b in blocks if b.region == Region.TOOL_SCHEMA]
    assert len(schema_blocks) == 1
    assert schema_blocks[0].tool_name == "my_tool"
