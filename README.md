# ContextLens

> **py-spy / pprof, but for what's inside your prompt.**

[![CI](https://github.com/contextlens/contextlens/actions/workflows/ci.yml/badge.svg)](https://github.com/contextlens/contextlens/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A diagnostic profiler for LLM agent context windows. It does **not** optimize or compress anything — it makes context waste **visible and quantified** so you can act on it.

<!-- Replace the line below with an actual screenshot/GIF after running `python examples/demo.py` and opening demo_report.html -->
![ContextLens treemap report](docs/treemap_placeholder.png)

---

## The Problem

In multi-turn agent loops, **the full context is re-sent on every API call**. A tool result added at turn 3 gets re-billed at turns 4, 5, 6 … Most of that is never read again.

Existing observability tools report a *total* token count — but never the **composition** or the **waste**. This invisible bloat drives three failures:

| Failure | Root cause |
|---------|-----------|
| **Cost** | You pay repeatedly for dead weight sitting in context |
| **Latency** | Fatter context means slower first-token time on every call |
| **Quality** | "Context rot" — models degrade as the window fills with stale, irrelevant material |

ContextLens is the **flamegraph** for this: it decomposes the window, shows re-billing over turns, detects specific waste patterns, and prints the **dollar cost** of each with a **concrete, one-line fix**.

---

## Install

```bash
pip install contextlens
```

Or from source:

```bash
git clone https://github.com/contextlens/contextlens
cd contextlens
pip install -e ".[dev]"
```

**Requirements:** Python 3.11+. No API key required for analysis.

---

## Quickstart — no API key needed

```bash
python examples/demo.py
```

This simulates a 30-turn agent loop with canned data, prints a ranked waste report to the terminal, and writes `examples/demo_report.html`. Open that file in any browser to see the interactive D3 treemap.

---

## CLI

```bash
# Terminal waste report
contextlens analyze trace.json

# Interactive HTML treemap report
contextlens report trace.json -o report.html
```

Example terminal output:

```
+---------------------------------------------------------------------+
| ContextLens | Run demo-001                                          |
| Model: claude-3-5-sonnet-20241022  | Provider: anthropic | Turns: 30 |
+---------------------------------------------------------------------+

  Context Composition by Region
  ---------------------------------------------------------------
  Region              Tokens    Cost (USD)   Share
  assistant_message   11,490    $0.0345      ###....... 25.5%
  tool_result         10,333    $0.0310      ##........ 22.9%
  tool_schema          9,450    $0.0284      ##........ 21.0%
  retrieved_content    5,805    $0.0174      #......... 12.9%
  user_message         4,740    $0.0142      #......... 10.5%
  system               3,240    $0.0097       #.........  7.2%
  TOTAL               45,058    $0.1352

  Re-billing: 45,058 tokens across 30 turns -> 43,185 (95.8%) re-billing waste ($0.1296)

  Top Waste Findings
  #   Type                Sev.    Wasted Tokens  Cost      Fix (truncated)
  1   [D]  duplicate      medium      7,084     $0.0213   Cache or externalize this content...
  2   [R]  redundant_ret  medium      5,805     $0.0174   Use a re-ranker or tighter threshold...
  3   [U]  unused_schema  low         3,150     $0.0095   Remove 'send_email' or inject dynamically...
  ...
```

---

## Python API

### Analyze a saved trace file

```python
import contextlens as cl

report = cl.analyze_file("trace.json")

print(f"Billed:      {report.total_tokens_billed:,} tokens  (${report.total_cost_usd:.4f})")
print(f"Recoverable: {report.recoverable_tokens:,} tokens  (${report.recoverable_cost_usd:.4f})")

for finding in report.findings_by_severity():
    print(f"[{finding.severity.upper():6}] {finding.kind.value:20} "
          f"{finding.wasted_tokens:>7,} tok  ${finding.wasted_cost_usd:.4f}")
    print(f"         Fix: {finding.fix}")

# Write the interactive HTML treemap
html = cl.render_html_report(report)
with open("report.html", "w") as f:
    f.write(html)
```

### Live capture — Anthropic

```python
import anthropic
import contextlens as cl

client = anthropic.Anthropic()

with cl.capture_anthropic(client, model="claude-3-5-sonnet-20241022") as collector:
    for turn in range(20):
        client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system="You are a helpful assistant.",
            messages=build_messages(turn),  # your agent loop
        )

trace = collector.build_trace()
collector.save("trace.json")          # optional: persist for later
report = cl.analyze_trace(trace)
```

### Live capture — OpenAI

```python
import openai
import contextlens as cl

client = openai.OpenAI()

with cl.capture_openai(client, model="gpt-4o") as collector:
    for turn in range(20):
        client.chat.completions.create(
            model="gpt-4o",
            messages=build_messages(turn),
        )

trace = collector.build_trace()
report = cl.analyze_trace(trace)
```

### Offline ingestion from existing logs

```python
import contextlens as cl

# If you already log your LLM requests as JSON dicts:
trace = cl.load_trace("my_existing_trace.json")
report = cl.analyze_trace(trace)
```

---

## Architecture

```
  +-----------------+      +-----------------+
  | Your Agent Loop |      | Saved JSON Trace|
  |  (Anthropic /   |      |  (existing logs)|
  |   OpenAI SDK)   |      +-----------------+
  +--------+--------+               |
           |                        |
  capture_anthropic()          load_trace()
  capture_openai()                  |
           |                        |
           v                        v
  +--------------------+   +--------------------+
  |   TraceCollector   |-->|       Trace        |
  |  (monkey-patches   |   | run_id, model,     |
  |   SDK client)      |   | provider,          |
  +--------------------+   | List[TurnSnapshot] |
                           +--------+-----------+
                                    |
                         +----------v----------+
                         |   DECOMPOSE         |
                         |  decompose.py       |
                         |                     |
                         | Classifies every    |
                         | content block into  |
                         | a Region:           |
                         |  - SYSTEM           |
                         |  - TOOL_SCHEMA      |
                         |  - TOOL_RESULT      |
                         |  - USER_MESSAGE     |
                         |  - ASSISTANT_MSG    |
                         |  - RETRIEVED_CONTENT|
                         |                     |
                         | Uses SHA-256 hash   |
                         | for cross-turn      |
                         | block identity      |
                         +----------+----------+
                                    |
              +---------------------+---------------------+
              |                                           |
   +----------v----------+              +----------------v-----------+
   |   RE-BILLING        |              |   WASTE DETECTORS          |
   |   rebilling.py      |              |   detectors.py             |
   |                     |              |                            |
   | Groups blocks by    |              | 1. DUPLICATE               |
   | content_hash across |              |    Same block re-sent      |
   | all turns.          |              |    verbatim N turns        |
   |                     |              |                            |
   | Per block:          |              | 2. NEAR_DUPLICATE          |
   |  - turns_present    |              |    Jaccard > 0.85 between  |
   |  - cumul_tokens     |              |    distinct blocks         |
   |  - cumul_cost_usd   |              |                            |
   |                     |              | 3. STALE_TOOL_RESULT       |
   | Recoverable waste = |              |    Tool output never       |
   | token*(turns-1)     |              |    referenced in later     |
   | for every block     |              |    assistant message       |
   | seen > 1 turn       |              |                            |
   +----------+----------+              | 4. UNUSED_TOOL_SCHEMA      |
              |                         |    Tool defined every turn |
              |                         |    but never called        |
              |                         |                            |
              |                         | 5. REDUNDANT_RETRIEVAL     |
              |                         |    Chunk overlap < 15%     |
              |                         |    with model output       |
              |                         +----------------+-----------+
              |                                          |
              +------------------+-----------------------+
                                 |
                      +----------v----------+
                      |    analyzer.py      |
                      |                     |
                      |  Builds Report:     |
                      |  - region_summaries |
                      |  - rebilling_entries|
                      |  - findings         |
                      |  - total costs      |
                      |  - recoverable $$   |
                      +----------+----------+
                                 |
              +------------------+------------------+
              |                  |                  |
   +----------v-----+  +---------v------+  +--------v-------+
   |  CLI           |  | HTML Report    |  | Python Report  |
   |  cli.py        |  | reporter.py    |  | Object (API)   |
   |                |  |                |  |                |
   | contextlens    |  | Single .html   |  | report.findings|
   | analyze        |  | file, no server|  | report.rebilling|
   |                |  | D3 treemap +   |  | render_html()  |
   | contextlens    |  | stacked area   |  |                |
   | report -o x    |  | + findings tbl |  |                |
   +----------------+  +----------------+  +----------------+
```

### Module map

| Module | Responsibility |
|--------|---------------|
| `models.py` | All dataclasses: `Trace`, `TurnSnapshot`, `ContentBlock`, `Region`, `WasteKind`, `Finding`, `Report` |
| `costs.py` | Pricing table for Anthropic + OpenAI models; `CostModel` with per-million USD rates |
| `tokenizer.py` | tiktoken for OpenAI (exact); char/4 heuristic for Anthropic (labeled approximation) |
| `capture.py` | Live SDK interception via context managers; `load_trace()` for offline JSON |
| `decompose.py` | Classifies raw request payloads into `ContentBlock` lists per turn; handles both Anthropic and OpenAI message schemas |
| `rebilling.py` | Groups blocks by content hash, computes cumulative re-billing cost, calculates recoverable waste |
| `detectors.py` | Four waste heuristics → `Finding` objects with severity, token count, USD cost, fix |
| `analyzer.py` | Orchestration: decompose → rebilling → detectors → `Report` |
| `cli.py` | Click CLI with Rich terminal output |
| `reporter.py` | Self-contained HTML report with inlined D3 v7 treemap + timeline |

### Data flow (one turn)

```
raw_request dict (messages, system, tools)
         |
         | decompose_snapshot()
         v
  List[ContentBlock]
    block_id  : uuid12
    region    : Region.TOOL_RESULT
    content   : "{'status': 'ok', ...}"
    token_count: 142
    content_hash: "sha256:abc..."
    tool_call_id: "tu_007"
         |
         | grouped across turns by content_hash
         v
  RebillingEntry
    turns_present: 18
    cumulative_tokens: 142 * 18 = 2556
    cumulative_cost_usd: $0.0077
         |
         | compared against later assistant blocks
         v
  Finding (STALE_TOOL_RESULT, severity=medium)
    wasted_tokens: 2556
    wasted_cost_usd: $0.0077
    fix: "Summarize immediately, drop raw result from context"
```

### Content region classification rules

| Rule | Region assigned |
|------|----------------|
| `system` parameter | `SYSTEM` |
| Items in `tools[]` / `functions[]` | `TOOL_SCHEMA` |
| Block with `type: tool_result` / role `tool` | `TOOL_RESULT` |
| Block with `type: tool_use` / has `tool_calls` | `ASSISTANT_MESSAGE` |
| Role `user`, text matches retrieval heuristic | `RETRIEVED_CONTENT` |
| Role `user`, plain text | `USER_MESSAGE` |
| Role `assistant`, no tool calls | `ASSISTANT_MESSAGE` |

The retrieval heuristic fires when text is > 200 chars and the first 500 chars contain a marker such as `retrieved:`, `chunk:`, `source:`, `document:`, `passage:`, `excerpt:`.

---

## Waste Detectors

### 1. Duplicate (`DUPLICATE`)
Exact content match (SHA-256) across > 1 turn. Wasted tokens = `token_count × (turns - 1)`.

**Fix:** Cache this content in the system prompt, use KV-cache-friendly structure, or send a compressed summary after the first use.

### 2. Near-Duplicate (`NEAR_DUPLICATE`)
Pairs of unique blocks with Jaccard word-4-gram similarity > 0.85. Only checks blocks ≥ 50 tokens.

**Fix:** Consolidate into a single template with variable slots.

### 3. Stale Tool Result (`STALE_TOOL_RESULT`)
Tool result block whose keyword set has < 2 words in common with any assistant message from the same turn onward.

**Fix:** Immediately after the tool call, have the assistant emit a short summary, then drop the raw result from context on the next turn.

### 4. Unused Tool Schema (`UNUSED_TOOL_SCHEMA`)
Tool defined in every turn's `tools[]` / `functions[]` array but with zero calls recorded across the entire trace.

**Fix:** Remove the schema, or inject it only when the agent enters the sub-flow that needs it.

### 5. Redundant Retrieval (`REDUNDANT_RETRIEVAL`)
Retrieval chunk classified as `RETRIEVED_CONTENT` whose keyword overlap with all subsequent assistant messages is < 15%, for chunks > 100 tokens.

**Fix:** Apply a re-ranker or raise the similarity score threshold before injecting chunks.

---

## Cost Model

Default prices (USD, mid-2025) are in `costs.py`. Override globally or per-analysis:

```python
from contextlens import CostModel, ModelPricing, analyze_trace

cm = CostModel(overrides={
    "my-internal-model": ModelPricing(input_per_million=0.50, output_per_million=1.50),
})
report = analyze_trace(trace, cost_model=cm)
```

---

## Token Counting

| Provider | Method | Notes |
|----------|--------|-------|
| OpenAI / GPT | `tiktoken` (exact) | Uses `encoding_for_model`; falls back to `cl100k_base` |
| Anthropic / Claude | `len(text) // 4` | Labeled approximation; ±10–15% typical error |
| Unknown | `len(text) // 4` | Same fallback |

---

## Roadmap

- [ ] Anthropic prompt caching awareness (cache breakpoint markers → deduct cached token cost)
- [ ] Per-turn diff view: what changed between turn N and N+1 highlighted in the HTML
- [ ] LangChain / LlamaIndex trace adapters
- [ ] Token budget watch mode: `contextlens watch --max-tokens 50000 --alert-pct 80`
- [ ] Gemini / Vertex AI provider support
- [ ] Export findings to OpenTelemetry spans / OTLP
- [ ] VS Code extension (annotate source with per-call cost)

---

## Contributing

```bash
git clone https://github.com/contextlens/contextlens
cd contextlens
pip install -e ".[dev]"

# Develop
ruff check src/ tests/       # lint
ruff format src/ tests/      # format
mypy src/contextlens/        # type check
pytest                        # tests
python examples/demo.py       # end-to-end smoke test
```

**Hard constraint:** ContextLens diagnoses — it does **not** compress, modify, or optimize prompts. Please keep that scope discipline in PRs.

Open issues, report bugs, or discuss the roadmap at [GitHub Issues](https://github.com/contextlens/contextlens/issues).

---

## License

[MIT](LICENSE) — Copyright (c) 2024 ContextLens Contributors
