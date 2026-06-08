---
title: ContextLens
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: true
license: mit
short_description: Diagnostic profiler for LLM agent context windows
tags:
  - llm
  - agents
  - observability
  - cost
  - tokens
  - context-window
  - diagnostics
---

# ContextLens

> **py-spy / pprof, but for what's inside your prompt.**

A diagnostic profiler for LLM agent context windows. Makes invisible context waste **visible and quantified** — so you can act on it.

## What it does

In multi-turn agent loops, the **full context is re-sent on every API call**. A tool result added at turn 3 gets re-billed at turns 4, 5, 6 … ContextLens decomposes the window, tracks re-billing cost per block across turns, and surfaces waste as ranked findings with dollar costs and one-line fixes.

### Five waste detectors

| Detector | What it finds |
|----------|--------------|
| **Duplicate** | Same block re-sent verbatim across multiple turns |
| **Near-Duplicate** | >85% Jaccard similarity between distinct blocks |
| **Stale Tool Result** | Tool output never referenced by a later assistant message |
| **Unused Tool Schema** | Tool defined every turn but never called |
| **Redundant Retrieval** | Retrieved chunk with <15% overlap with model output |

## How to use this Space

**Tab 1 — Live Demo:** Click "Run Demo Analysis" to analyze a simulated 30-turn agent loop (JWT migration task). No file or API key needed. Fires all five detectors.

**Tab 2 — Analyze Your Trace:** Upload a `trace.json` captured from your own agent run using the Python SDK:

```python
pip install contextlens

import contextlens as cl
with cl.capture_anthropic(client, model="claude-3-5-sonnet-20241022") as collector:
    # ... your agent loop ...
collector.save("trace.json")
```

Then upload the saved file here.

## Links

- **GitHub:** https://github.com/HarshalSant/contextlens
- **Install:** `pip install contextlens`
- **License:** MIT
