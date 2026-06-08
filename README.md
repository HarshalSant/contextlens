# ContextLens

> **py-spy / pprof, but for what's inside your prompt.**

A diagnostic profiler for LLM agent context windows. ContextLens makes invisible context waste visible and quantified — so you can act on it.

![treemap placeholder](docs/treemap_placeholder.png)
<!-- Replace with actual screenshot/GIF after running `python examples/demo.py` -->

---

## The Problem

In multi-turn agent loops, the **full context is re-sent on every API call**. A tool result added at turn 3 gets re-billed at turns 4, 5, 6… Most of that is never read again.

Existing tools report a *total* token count — but never the **composition** or the **waste**. This invisible bloat drives three failures:

| Failure | Cause |
|---------|-------|
| **Cost** | You pay repeatedly for dead weight |
| **Latency** | Fatter context → slower responses |
| **Quality** | "Context rot" — models degrade as the window fills with stale, irrelevant material |

ContextLens is the **flamegraph** for this: it decomposes the window, shows re-billing over time, detects specific waste patterns, and prints the **dollar cost** of each with a **concrete fix**.

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

**Requirements:** Python 3.11+, no API key needed for analysis.

---

## Quickstart (no API key required)

Run the built-in demo — simulates a 30-turn agent loop and generates a full report:

```bash
python examples/demo.py
```

This prints a ranked waste report to the terminal **and** writes `examples/demo_report.html`. Open that file in any browser to see the interactive treemap.

---

## CLI Usage

```bash
# Terminal report
contextlens analyze examples/demo_trace.json

# Generate interactive HTML treemap
contextlens report examples/demo_trace.json -o report.html
```

Example terminal output:

```
╭─────────────────────────────────────────────────╮
│ ContextLens — Run demo-001                      │
│ Model: claude-3-5-sonnet-20241022  Turns: 30    │
╰─────────────────────────────────────────────────╯

 Context Composition by Region
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Region              Tokens    Cost (USD)  Share
  tool_result         42,180    $0.1265     ████████░░  38.2%
  assistant_message   28,450    $0.0854     ██████░░░░  25.8%
  tool_schema         18,600    $0.0558     ████░░░░░░  16.9%
  system               8,200    $0.0246     ██░░░░░░░░   7.4%
  user_message         7,120    $0.0214     █░░░░░░░░░   6.5%
  TOTAL              110,450    $0.3314

 Re-billing: 110,450 tokens billed across 30 turns
 → 38,200 tokens (34.6%) are re-billing waste ($0.1146)

 Top Waste Findings
 #  Type              Sev.    Wasted Tokens  Cost      Description
 1  [U] unused_schema high    15,600         $0.0468   Tool 'send_email' defined 30 turns, never called
 2  [D] duplicate     high    12,400         $0.0372   Block 'Please continue working on auth…' re-sent 8×
 3  [S] stale_result  medium   6,300         $0.0189   Tool result for 'search_code' never referenced
 ...
```

---

## Python API

```python
import contextlens as cl

# Analyze a saved trace file
report = cl.analyze_file("trace.json")

print(f"Total billed: {report.total_tokens_billed:,} tokens (${report.total_cost_usd:.4f})")
print(f"Recoverable:  {report.recoverable_tokens:,} tokens (${report.recoverable_cost_usd:.4f})")

for finding in report.findings_by_severity():
    print(f"[{finding.severity.upper()}] {finding.kind.value}: {finding.wasted_tokens:,} tok — {finding.fix}")

# Generate HTML report
html = cl.render_html_report(report)
with open("report.html", "w") as f:
    f.write(html)
```

### Live capture (Anthropic)

```python
import anthropic
import contextlens as cl

client = anthropic.Anthropic()

with cl.capture_anthropic(client, model="claude-3-5-sonnet-20241022") as collector:
    for _ in range(10):
        client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[...],
        )

trace = collector.build_trace()
report = cl.analyze_trace(trace)
```

### Live capture (OpenAI)

```python
import openai
import contextlens as cl

client = openai.OpenAI()

with cl.capture_openai(client, model="gpt-4o") as collector:
    for _ in range(10):
        client.chat.completions.create(model="gpt-4o", messages=[...])

trace = collector.build_trace()
report = cl.analyze_trace(trace)
```

---

## How It Works

```
 Raw LLM calls
      │
      ▼
  ┌─────────┐    capture_anthropic / capture_openai
  │ Capture │ ── or load_trace("trace.json")
  └────┬────┘
       │  Trace (TurnSnapshot × N)
       ▼
  ┌──────────┐
  │ Decompose│  classifies every content block → Region
  └────┬─────┘  (system, tool_schema, tool_result, user_msg, asst_msg, retrieved)
       │
       ▼
  ┌───────────┐
  │ Re-billing│  tracks each block across turns, computes cumulative cost
  └────┬──────┘
       │
       ▼
  ┌──────────┐
  │ Detectors│  four heuristics: duplicate, stale_result, unused_schema, redundant_retrieval
  └────┬─────┘
       │
       ▼
  ┌────────┐
  │ Report │  structured Report object + CLI print + HTML treemap
  └────────┘
```

### Waste detectors

| Detector | What it finds | Fix |
|----------|---------------|-----|
| **Duplicate** | Same content re-sent verbatim across turns | Cache in system prompt or use KV cache |
| **Near-duplicate** | >85% Jaccard similarity between blocks | Consolidate into a template |
| **Stale tool result** | Tool output never referenced by later assistant | Summarize immediately, drop raw result |
| **Unused tool schema** | Tool defined every turn but never called | Remove or inject dynamically |
| **Redundant retrieval** | Retrieval chunk with <15% overlap with model output | Re-rank before injecting |

---

## Roadmap

- [ ] Prompt caching awareness (Anthropic cache breakpoints, OpenAI cached tokens)
- [ ] Per-turn diff view: what changed between turn N and N+1
- [ ] LangChain / LlamaIndex trace adapters
- [ ] Token budget alerts (`contextlens watch --max-tokens 50000`)
- [ ] Gemini provider support
- [ ] Export to OpenTelemetry spans

---

## Contributing

```bash
git clone https://github.com/contextlens/contextlens
cd contextlens
pip install -e ".[dev]"
ruff check src/          # lint
mypy src/contextlens/    # type check
pytest                   # tests
```

PRs welcome. Please keep the scope discipline: ContextLens **diagnoses** — it does not compress, optimize, or modify prompts. Keep that as a hard constraint.

---

## License

MIT — see [LICENSE](LICENSE).
