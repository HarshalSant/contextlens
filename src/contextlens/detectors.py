"""Waste detection heuristics.

Four detectors, each returning a list of Finding objects:
1. Duplicate / near-duplicate content across turns.
2. Stale tool results (present but never referenced later).
3. Tool schemas defined but never called.
4. Redundant retrieval chunks (low overlap with model output).
"""

from __future__ import annotations

import re
from collections import defaultdict

from .costs import CostModel, default_cost_model
from .models import ContentBlock, Finding, Region, Trace, WasteKind


def run_all_detectors(
    trace: Trace,
    cost_model: CostModel | None = None,
) -> list[Finding]:
    cm = cost_model or default_cost_model
    findings: list[Finding] = []
    findings.extend(detect_duplicates(trace, cm))
    findings.extend(detect_stale_tool_results(trace, cm))
    findings.extend(detect_unused_tool_schemas(trace, cm))
    findings.extend(detect_redundant_retrieval(trace, cm))
    return findings


# ---------------------------------------------------------------------------
# 1. Duplicate / near-duplicate content
# ---------------------------------------------------------------------------


def detect_duplicates(trace: Trace, cm: CostModel) -> list[Finding]:
    """Find content blocks that appear verbatim on multiple turns."""
    findings: list[Finding] = []
    model = trace.model

    # Exact duplicates: same content_hash appearing on > 1 turn
    hash_to_blocks: dict[str, list[ContentBlock]] = defaultdict(list)
    for turn in trace.turns:
        for block in turn.blocks:
            if block.content_hash and block.region not in (Region.TOOL_SCHEMA,):
                hash_to_blocks[block.content_hash].append(block)

    for _content_hash, blocks in hash_to_blocks.items():
        turns_with_block = sorted({b.turn_index for b in blocks})
        if len(turns_with_block) <= 1:
            continue

        representative = blocks[0]
        wasted_turns = len(turns_with_block) - 1
        wasted_tokens = representative.token_count * wasted_turns
        wasted_cost = cm.input_cost(model, wasted_tokens)

        severity = _severity_from_cost(wasted_cost)
        preview = representative.content[:80].replace("\n", " ")

        findings.append(
            Finding(
                kind=WasteKind.DUPLICATE,
                severity=severity,
                description=(
                    f"Block '{preview}…' ({representative.token_count} tok) "
                    f"re-sent verbatim on {len(turns_with_block)} turns "
                    f"(first: {turns_with_block[0]}, last: {turns_with_block[-1]})"
                ),
                fix=(
                    "Cache or externalize this content and pass a reference instead; "
                    "use a system-prompt slot or KV-cache-friendly structure."
                ),
                wasted_tokens=wasted_tokens,
                wasted_cost_usd=wasted_cost,
                affected_block_ids=[b.block_id for b in blocks],
                first_seen_turn=turns_with_block[0],
                last_seen_turn=turns_with_block[-1],
            )
        )

    # Near-duplicates: Jaccard similarity > 0.85 between distinct blocks
    findings.extend(_detect_near_duplicates(trace, cm))

    return findings


def _shingle(text: str, k: int = 4) -> set[str]:
    """Character k-shingle set for Jaccard similarity."""
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _detect_near_duplicates(trace: Trace, cm: CostModel) -> list[Finding]:
    findings: list[Finding] = []
    model = trace.model
    THRESHOLD = 0.85
    MIN_TOKENS = 50  # Only check substantive blocks

    # Collect unique content blocks (by hash) that are large enough
    seen_hashes: set[str] = set()
    candidates: list[ContentBlock] = []
    for turn in trace.turns:
        for block in turn.blocks:
            if (
                block.token_count >= MIN_TOKENS
                and block.content_hash not in seen_hashes
                and block.region not in (Region.TOOL_SCHEMA, Region.SYSTEM)
            ):
                seen_hashes.add(block.content_hash)
                candidates.append(block)

    # O(n^2) pairwise — acceptable for typical trace sizes (< 500 unique blocks)
    reported_pairs: set[frozenset[str]] = set()
    for i, a in enumerate(candidates):
        shingles_a = _shingle(a.content)
        for b in candidates[i + 1 :]:
            pair_key = frozenset([a.content_hash, b.content_hash])
            if pair_key in reported_pairs:
                continue
            shingles_b = _shingle(b.content)
            sim = _jaccard(shingles_a, shingles_b)
            if sim >= THRESHOLD:
                reported_pairs.add(pair_key)
                wasted_tokens = min(a.token_count, b.token_count)
                wasted_cost = cm.input_cost(model, wasted_tokens)
                findings.append(
                    Finding(
                        kind=WasteKind.NEAR_DUPLICATE,
                        severity=_severity_from_cost(wasted_cost),
                        description=(
                            f"Near-duplicate blocks (Jaccard={sim:.2f}): "
                            f"turn {a.turn_index} ({a.token_count} tok) ≈ "
                            f"turn {b.turn_index} ({b.token_count} tok)"
                        ),
                        fix=(
                            "Consolidate overlapping content into a single block; "
                            "consider a template with a variable slot instead of re-constructing."
                        ),
                        wasted_tokens=wasted_tokens,
                        wasted_cost_usd=wasted_cost,
                        affected_block_ids=[a.block_id, b.block_id],
                        first_seen_turn=min(a.turn_index, b.turn_index),
                        last_seen_turn=max(a.turn_index, b.turn_index),
                    )
                )

    return findings


# ---------------------------------------------------------------------------
# 2. Stale tool results
# ---------------------------------------------------------------------------


def detect_stale_tool_results(trace: Trace, cm: CostModel) -> list[Finding]:
    """Detect tool result blocks that are never referenced by a later assistant turn."""
    findings: list[Finding] = []
    model = trace.model

    # Gather all tool results with the turn they first appear
    tool_results: list[ContentBlock] = []
    for turn in trace.turns:
        for block in turn.blocks:
            if block.region == Region.TOOL_RESULT:
                tool_results.append(block)

    if not tool_results:
        return findings

    # Build a set of keywords from each tool result
    # then check if any later assistant block overlaps
    for tr_block in tool_results:
        keywords = _extract_keywords(tr_block.content)
        if not keywords:
            continue

        referenced = False
        for turn in trace.turns:
            if turn.turn_index < tr_block.turn_index:
                continue
            for block in turn.blocks:
                if block.region == Region.ASSISTANT_MESSAGE:
                    block_keywords = _extract_keywords(block.content)
                    overlap = keywords & block_keywords
                    if len(overlap) >= 2:
                        referenced = True
                        break
            if referenced:
                break

        if not referenced:
            # Count how many turns it persisted (how many later turns carry it)
            later_turns_with_result = sum(
                1
                for t in trace.turns
                if t.turn_index > tr_block.turn_index
                and any(b.content_hash == tr_block.content_hash for b in t.blocks)
            )
            wasted_tokens = tr_block.token_count * max(1, later_turns_with_result)
            wasted_cost = cm.input_cost(model, wasted_tokens)

            tool_label = tr_block.tool_name or tr_block.tool_call_id or "unknown"
            findings.append(
                Finding(
                    kind=WasteKind.STALE_TOOL_RESULT,
                    severity=_severity_from_cost(wasted_cost),
                    description=(
                        f"Tool result for '{tool_label}' ({tr_block.token_count} tok) "
                        f"introduced at turn {tr_block.turn_index} but never referenced "
                        f"by any subsequent assistant message."
                    ),
                    fix=(
                        "Summarize this result into a shorter assistant message immediately "
                        "after the tool call, then drop the raw result from context."
                    ),
                    wasted_tokens=wasted_tokens,
                    wasted_cost_usd=wasted_cost,
                    affected_block_ids=[tr_block.block_id],
                    first_seen_turn=tr_block.turn_index,
                    last_seen_turn=trace.turns[-1].turn_index
                    if trace.turns
                    else tr_block.turn_index,
                )
            )

    return findings


# ---------------------------------------------------------------------------
# 3. Unused tool schemas
# ---------------------------------------------------------------------------


def detect_unused_tool_schemas(trace: Trace, cm: CostModel) -> list[Finding]:
    """Detect tool schemas that are defined in every turn but never called."""
    findings: list[Finding] = []
    model = trace.model

    if not trace.turns:
        return findings

    # Tools called anywhere in the trace
    called_anywhere: set[str] = set()
    for turn in trace.turns:
        called_anywhere.update(turn.tool_names_called)

    # Tools defined in every turn
    tools_always_defined: set[str] | None = None
    for turn in trace.turns:
        defined = set(turn.tool_names_defined)
        if tools_always_defined is None:
            tools_always_defined = defined
        else:
            tools_always_defined &= defined

    if not tools_always_defined:
        return findings

    never_called = tools_always_defined - called_anywhere

    for tool_name in never_called:
        # Find representative schema block
        schema_blocks = [
            block
            for turn in trace.turns
            for block in turn.blocks
            if block.region == Region.TOOL_SCHEMA and block.tool_name == tool_name
        ]
        if not schema_blocks:
            continue

        representative = schema_blocks[0]
        wasted_tokens = representative.token_count * len(trace.turns)
        wasted_cost = cm.input_cost(model, wasted_tokens)

        findings.append(
            Finding(
                kind=WasteKind.UNUSED_TOOL_SCHEMA,
                severity=_severity_from_cost(wasted_cost),
                description=(
                    f"Tool '{tool_name}' ({representative.token_count} tok/turn) "
                    f"defined across {len(trace.turns)} turns but never called."
                ),
                fix=(
                    f"Remove '{tool_name}' from the tool list, or add it only "
                    "when it becomes relevant (dynamic tool injection)."
                ),
                wasted_tokens=wasted_tokens,
                wasted_cost_usd=wasted_cost,
                affected_block_ids=[b.block_id for b in schema_blocks],
                first_seen_turn=0,
                last_seen_turn=trace.turns[-1].turn_index,
            )
        )

    return findings


# ---------------------------------------------------------------------------
# 4. Redundant retrieval chunks
# ---------------------------------------------------------------------------


def detect_redundant_retrieval(trace: Trace, cm: CostModel) -> list[Finding]:
    """Detect retrieved chunks with low overlap with subsequent assistant output."""
    findings: list[Finding] = []
    model = trace.model

    for turn in trace.turns:
        retrieval_blocks = [b for b in turn.blocks if b.region == Region.RETRIEVED_CONTENT]
        if not retrieval_blocks:
            continue

        # Keywords from assistant turns AFTER this retrieval turn
        later_assistant_text = " ".join(
            block.content
            for later_turn in trace.turns
            if later_turn.turn_index > turn.turn_index
            for block in later_turn.blocks
            if block.region == Region.ASSISTANT_MESSAGE
        )
        later_keywords = _extract_keywords(later_assistant_text)

        for block in retrieval_blocks:
            chunk_keywords = _extract_keywords(block.content)
            if not chunk_keywords:
                continue

            overlap = chunk_keywords & later_keywords
            overlap_ratio = len(overlap) / len(chunk_keywords) if chunk_keywords else 0.0

            if overlap_ratio < 0.15 and block.token_count > 100:
                # Count how many future turns carry this chunk
                future_turns = sum(
                    1
                    for t in trace.turns
                    if t.turn_index >= turn.turn_index
                    and any(b.content_hash == block.content_hash for b in t.blocks)
                )
                wasted_tokens = block.token_count * future_turns
                wasted_cost = cm.input_cost(model, wasted_tokens)

                findings.append(
                    Finding(
                        kind=WasteKind.REDUNDANT_RETRIEVAL,
                        severity=_severity_from_cost(wasted_cost),
                        description=(
                            f"Retrieval chunk at turn {turn.turn_index} "
                            f"({block.token_count} tok, overlap={overlap_ratio:.1%}) "
                            "has low overlap with subsequent assistant output — "
                            "likely never used."
                        ),
                        fix=(
                            "Use a re-ranker or tighter similarity threshold to filter "
                            "low-relevance chunks before adding them to context."
                        ),
                        wasted_tokens=wasted_tokens,
                        wasted_cost_usd=wasted_cost,
                        affected_block_ids=[block.block_id],
                        first_seen_turn=turn.turn_index,
                        last_seen_turn=trace.turns[-1].turn_index
                        if trace.turns
                        else turn.turn_index,
                    )
                )

    return findings


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "from",
        "up",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "each",
        "than",
        "so",
        "but",
        "and",
        "or",
        "not",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "them",
        "their",
        "its",
        "his",
        "her",
        "our",
        "your",
        "my",
        "what",
        "which",
        "who",
        "when",
        "where",
        "how",
        "all",
        "any",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "as",
        "very",
        "just",
        "because",
        "if",
        "while",
        "since",
        "although",
        "though",
    ]
)


def _extract_keywords(text: str, min_len: int = 4) -> set[str]:
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return {w for w in words if len(w) >= min_len and w not in _STOP_WORDS}


def _severity_from_cost(cost_usd: float) -> str:
    if cost_usd >= 0.10:
        return "high"
    if cost_usd >= 0.01:
        return "medium"
    return "low"
