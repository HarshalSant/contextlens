"""Decompose raw LLM request payloads into typed ContentBlocks."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from .models import ContentBlock, Region, TurnSnapshot
from .tokenizer import RETRIEVAL_MARKERS, count_tokens


def decompose_trace_turns(turns: list[TurnSnapshot]) -> list[TurnSnapshot]:
    """Populate `blocks` for every turn in-place and return the list."""
    for snap in turns:
        snap.blocks = decompose_snapshot(snap)
    return turns


def decompose_snapshot(snap: TurnSnapshot) -> list[ContentBlock]:
    """Extract ContentBlocks from a TurnSnapshot's raw_request."""
    req = snap.raw_request
    provider = snap.provider.lower()

    if provider == "anthropic":
        return _decompose_anthropic(req, snap.turn_index, snap.model)
    elif provider == "openai":
        return _decompose_openai(req, snap.turn_index, snap.model)
    else:
        return _decompose_generic(req, snap.turn_index, snap.model)


# ---------------------------------------------------------------------------
# Anthropic decomposition
# ---------------------------------------------------------------------------


def _decompose_anthropic(
    req: dict[str, Any],
    turn_index: int,
    model: str,
) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []

    # System prompt
    system = req.get("system")
    if system:
        text = system if isinstance(system, str) else json.dumps(system)
        blocks.append(_make_block(Region.SYSTEM, text, turn_index, model))

    # Tool schemas
    for tool in req.get("tools", []):
        text = json.dumps(tool)
        b = _make_block(Region.TOOL_SCHEMA, text, turn_index, model)
        b.tool_name = tool.get("name")
        blocks.append(b)

    # Messages
    for msg in req.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        blocks.extend(_parse_anthropic_content(content, role, turn_index, model))

    return blocks


def _parse_anthropic_content(
    content: Any,
    role: str,
    turn_index: int,
    model: str,
) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []

    if isinstance(content, str):
        region = _role_to_region(role)
        if _looks_like_retrieval(content):
            region = Region.RETRIEVED_CONTENT
        blocks.append(_make_block(region, content, turn_index, model))
        return blocks

    if not isinstance(content, list):
        return blocks

    for item in content:
        if not isinstance(item, dict):
            continue
        block_type = item.get("type", "")

        if block_type == "text":
            text = item.get("text", "")
            region = _role_to_region(role)
            if _looks_like_retrieval(text):
                region = Region.RETRIEVED_CONTENT
            blocks.append(_make_block(region, text, turn_index, model))

        elif block_type == "tool_use":
            # Assistant is calling a tool
            text = json.dumps(item)
            b = _make_block(Region.ASSISTANT_MESSAGE, text, turn_index, model)
            b.tool_name = item.get("name")
            b.tool_call_id = item.get("id")
            blocks.append(b)

        elif block_type == "tool_result":
            result_content = item.get("content", "")
            if isinstance(result_content, list) or not isinstance(result_content, str):
                result_content = json.dumps(result_content)
            b = _make_block(Region.TOOL_RESULT, result_content, turn_index, model)
            b.tool_call_id = item.get("tool_use_id")
            blocks.append(b)

        else:
            text = json.dumps(item)
            blocks.append(_make_block(_role_to_region(role), text, turn_index, model))

    return blocks


# ---------------------------------------------------------------------------
# OpenAI decomposition
# ---------------------------------------------------------------------------


def _decompose_openai(
    req: dict[str, Any],
    turn_index: int,
    model: str,
) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []

    # Tool schemas (OpenAI wraps them in {"type": "function", "function": {...}})
    for tool in req.get("tools", []):
        text = json.dumps(tool)
        b = _make_block(Region.TOOL_SCHEMA, text, turn_index, model)
        fn = tool.get("function", {})
        b.tool_name = fn.get("name") if fn else None
        blocks.append(b)

    # Messages
    for msg in req.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        tool_call_id = msg.get("tool_call_id")

        if role == "system":
            text = content if isinstance(content, str) else json.dumps(content)
            blocks.append(_make_block(Region.SYSTEM, text, turn_index, model))

        elif role == "tool":
            # Tool result message
            text = content if isinstance(content, str) else json.dumps(content)
            b = _make_block(Region.TOOL_RESULT, text, turn_index, model)
            b.tool_call_id = tool_call_id
            blocks.append(b)

        elif role == "assistant" and tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                text = json.dumps(tc)
                b = _make_block(Region.ASSISTANT_MESSAGE, text, turn_index, model)
                b.tool_name = fn.get("name")
                b.tool_call_id = tc.get("id")
                blocks.append(b)
            # Also capture any text content alongside tool calls
            if content:
                text = content if isinstance(content, str) else json.dumps(content)
                blocks.append(_make_block(Region.ASSISTANT_MESSAGE, text, turn_index, model))

        elif role == "user":
            if isinstance(content, str):
                region = Region.RETRIEVED_CONTENT if _looks_like_retrieval(content) else Region.USER_MESSAGE
                blocks.append(_make_block(region, content, turn_index, model))
            elif isinstance(content, list):
                for part in content:
                    text = part.get("text", "") if isinstance(part, dict) else str(part)
                    region = Region.RETRIEVED_CONTENT if _looks_like_retrieval(text) else Region.USER_MESSAGE
                    blocks.append(_make_block(region, text, turn_index, model))

        elif role == "assistant":
            text = content if isinstance(content, str) else json.dumps(content)
            blocks.append(_make_block(Region.ASSISTANT_MESSAGE, text, turn_index, model))

    return blocks


# ---------------------------------------------------------------------------
# Generic fallback (for unknown providers)
# ---------------------------------------------------------------------------


def _decompose_generic(
    req: dict[str, Any],
    turn_index: int,
    model: str,
) -> list[ContentBlock]:
    text = json.dumps(req)
    return [_make_block(Region.UNKNOWN, text, turn_index, model)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_block(region: Region, content: str, turn_index: int, model: str) -> ContentBlock:
    token_count = count_tokens(content, model)
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    return ContentBlock(
        block_id=str(uuid.uuid4())[:12],
        region=region,
        content=content,
        token_count=token_count,
        turn_index=turn_index,
        first_seen_turn=turn_index,
        last_seen_turn=turn_index,
        content_hash=content_hash,
    )


def _role_to_region(role: str) -> Region:
    if role == "user":
        return Region.USER_MESSAGE
    if role == "assistant":
        return Region.ASSISTANT_MESSAGE
    return Region.UNKNOWN


def _looks_like_retrieval(text: str) -> bool:
    """Heuristic: does this text look like a retrieved document chunk?"""
    if len(text) < 200:
        return False
    return bool(RETRIEVAL_MARKERS.search(text[:500]))
