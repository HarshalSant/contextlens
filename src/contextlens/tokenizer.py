"""Token counting utilities for OpenAI (tiktoken) and Anthropic models."""

from __future__ import annotations

import re

_tiktoken_cache: dict[str, object] = {}


def _get_tiktoken_enc(model: str) -> object:
    if model not in _tiktoken_cache:
        try:
            import tiktoken

            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            _tiktoken_cache[model] = enc
        except ImportError:
            _tiktoken_cache[model] = None
    return _tiktoken_cache[model]


def count_tokens(text: str, model: str) -> int:
    """Return an approximate token count for *text* given *model*.

    - OpenAI models use tiktoken (exact).
    - Anthropic models use the chars/4 heuristic (labeled approximation).
    - Unknown models fall back to the heuristic.
    """
    if not text:
        return 0

    lower = model.lower()

    if any(p in lower for p in ("gpt-", "o1", "o3", "text-davinci", "text-embedding")):
        enc = _get_tiktoken_enc(model)
        if enc is not None:
            return len(enc.encode(text))  # type: ignore[attr-defined]

    # Anthropic / fallback: character-based approximation (≈4 chars per token)
    return max(1, len(text) // 4)


def count_messages_tokens(messages: list[dict[str, object]], model: str) -> int:
    """Approximate token count for a messages list (role + content pairs)."""
    total = 0
    for msg in messages:
        role = str(msg.get("role", ""))
        content = msg.get("content", "")
        total += count_tokens(role, model)
        if isinstance(content, str):
            total += count_tokens(content, model)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += count_tokens(str(block.get("text", block.get("content", ""))), model)
    # OpenAI overhead: 4 tokens per message + 2 reply priming
    overhead = len(messages) * 4 + 2
    return total + overhead


def count_tool_schema_tokens(tool: dict[str, object], model: str) -> int:
    """Count tokens for a single tool schema definition."""
    import json

    return count_tokens(json.dumps(tool), model)


# Regex used by decomposer to spot retrieval-chunk markers
RETRIEVAL_MARKERS = re.compile(
    r"(retrieved|chunk|document|passage|context|source|excerpt|snippet)\s*[:\-#\[]",
    re.IGNORECASE,
)
