"""ContextLens — Hugging Face Spaces Gradio app (Gradio 5 compatible)."""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

# Install the local package if running from the repo root (HF Spaces)
_root = Path(__file__).parent
if (_root / "src" / "contextlens").exists():
    sys.path.insert(0, str(_root / "src"))

import gradio as gr
import pandas as pd

# ---------------------------------------------------------------------------
# Lazy import with a friendly error if the package isn't available
# ---------------------------------------------------------------------------
try:
    from contextlens.analyzer import analyze_trace
    from contextlens.capture import _dict_to_trace
    from contextlens.models import Report
    from contextlens.reporter import render_html_report
    _IMPORT_OK = True
    _IMPORT_ERROR = ""
except Exception as e:
    _IMPORT_OK = False
    _IMPORT_ERROR = traceback.format_exc()


# ---------------------------------------------------------------------------
# Demo trace builder — inlined so we have ZERO dependency on examples/demo.py
# ---------------------------------------------------------------------------

import hashlib
import uuid
from datetime import datetime, timedelta


_SYSTEM_PROMPT = """\
You are a professional software engineering assistant helping a team build a SaaS application.
You have access to tools for searching code, reading files, running tests, and querying databases.
Always be precise, cite relevant code when possible, and suggest tests for any changes you make.
Follow the team's coding standards: Python 3.11+, typed, ruff-clean, pytest for tests.
Do not make breaking changes without explicit approval.\
""".strip()

_TOOL_SCHEMAS = [
    {
        "name": "search_code",
        "description": "Search the codebase for a pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "run_tests",
        "description": "Execute the test suite or a subset of tests.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "flags": {"type": "string"},
            },
        },
    },
    {
        "name": "query_database",
        "description": "Run a read-only SQL query against the production database.",
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    },
    # Defined every turn, never called → triggers unused_tool_schema detector
    {
        "name": "send_email",
        "description": "Send an email notification to a team member.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]

_STALE_RETRIEVAL = """\
retrieved document: Architecture Decision Record #42
source: docs/adr/0042-database-strategy.md
excerpt: We evaluated three database strategies for the user data store: PostgreSQL with
row-level security, MongoDB with document-level permissions, and a hybrid approach using
PostgreSQL for structured data and Redis for session caching. After a 2-week POC we selected
PostgreSQL with row-level security because it aligns with our existing operational expertise,
supports complex relational queries needed for billing, and integrates cleanly with SQLAlchemy.
Migration path: existing MySQL tables migrated via Alembic with zero-downtime blue-green deploys.
Performance target: p99 query latency < 50ms for all user-facing endpoints.\
""".strip()

_SEARCH_RESULT = """\
search_code result for pattern='class UserService':
  File: src/services/user_service.py, line 42
  class UserService:
      def __init__(self, db: Database, cache: RedisCache) -> None:
          self.db = db; self.cache = cache
      def get_user(self, user_id: int) -> User | None:
          cached = self.cache.get(f'user:{user_id}')
          if cached: return User.model_validate(cached)
          user = self.db.query(User).filter_by(id=user_id).first()
          if user: self.cache.set(f'user:{user_id}', user.model_dump(), ttl=300)
          return user
      def create_user(self, payload: CreateUserRequest) -> User:
          if self.db.query(User).filter_by(email=payload.email).first():
              raise DuplicateEmailError(payload.email)
          user = User(**payload.model_dump())
          self.db.add(user); self.db.commit()
          self.cache.invalidate('user:*'); return user\
""".strip()

_REPEAT_PREFIX = "Please continue working on the authentication module."


def _build_demo_trace() -> dict:
    base_time = datetime(2024, 6, 1, 10, 0, 0)
    turns = []
    messages: list[dict] = []

    steps = [
        ("I'll start by searching for the existing auth code.", ["search_code"]),
        ("I found the UserService. Let me read the auth implementation.", ["read_file"]),
        ("Now I have a clear picture. I'll design the JWT migration.", []),
        (f"{_STALE_RETRIEVAL}\n\n{_REPEAT_PREFIX}", []),
        ("Here is the JWTService implementation:\n```python\nclass JWTService:\n    def encode(self, user_id: int) -> str: ...\n    def decode(self, token: str) -> dict: ...\n```", ["run_tests"]),
        ("Running tests (iteration 1)… adjusting token expiry handling.", ["run_tests"]),
        ("Running tests (iteration 2)… fixing refresh token logic.", ["run_tests"]),
        ("Running tests (iteration 3)… edge case: expired token during refresh.", ["run_tests"]),
        ("All core tests pass. Adding middleware integration.", []),
        ("Test failure: refresh endpoint missing. Adding it now.", ["read_file"]),
        ("Adding refresh token endpoint.", ["run_tests"]),
        ("Updating middleware to validate JWT on every request.", ["search_code"]),
        ("Writing integration tests for the new auth flow.", ["query_database"]),
        ("Fixing edge case: expired token during refresh.", []),
        ("Updating documentation strings.", ["run_tests"]),
        ("Running full test suite.", ["search_code"]),
        ("Addressing review comments on the PR.", []),
        ("Squashing migrations into a single file.", ["run_tests"]),
        ("Final cleanup pass — removing dead session code.", []),
        ("All tests green. Preparing summary.", []),
    ]

    for i in range(30):
        ts = base_time + timedelta(minutes=i * 2)

        if i == 0:
            messages = [
                {"role": "user", "content": "Help me refactor the authentication module to use JWT tokens."},
                {"role": "assistant", "content": steps[0][0]},
            ]
            tool_calls = steps[0][1]
        elif i == 1:
            messages = messages + [
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_001", "content": _SEARCH_RESULT}]},
                {"role": "assistant", "content": steps[1][0]},
            ]
            tool_calls = steps[1][1]
        elif i < len(steps):
            messages = messages + [
                {"role": "user", "content": _REPEAT_PREFIX},
                {"role": "assistant", "content": steps[i][0]},
            ]
            tool_calls = steps[i][1]
        else:
            messages = messages + [
                {"role": "user", "content": _REPEAT_PREFIX},
                {"role": "assistant", "content": f"Wrap-up turn {i}: finalizing JWT auth migration. All changes merged."},
            ]
            tool_calls = []

        system_tok = len(_SYSTEM_PROMPT) // 4
        tools_tok = sum(len(json.dumps(t)) // 4 for t in _TOOL_SCHEMAS)
        msgs_tok = sum(len(json.dumps(m)) // 4 for m in messages)

        turns.append({
            "turn_index": i,
            "timestamp": ts.isoformat(),
            "model": "claude-3-5-sonnet-20241022",
            "provider": "anthropic",
            "total_tokens": system_tok + tools_tok + msgs_tok,
            "tool_names_defined": [t["name"] for t in _TOOL_SCHEMAS],
            "tool_names_called": tool_calls,
            "raw_request": {
                "model": "claude-3-5-sonnet-20241022",
                "system": _SYSTEM_PROMPT,
                "tools": _TOOL_SCHEMAS,
                "messages": list(messages),
            },
        })

    return {
        "run_id": "demo-001",
        "model": "claude-3-5-sonnet-20241022",
        "provider": "anthropic",
        "created_at": base_time.isoformat(),
        "turns": turns,
    }


# ---------------------------------------------------------------------------
# Report → Gradio components
# ---------------------------------------------------------------------------


def _summary_html(report: Report) -> str:
    rec_pct = (
        report.recoverable_tokens / report.total_tokens_billed * 100
        if report.total_tokens_billed else 0.0
    )
    n_high = sum(1 for f in report.findings if f.severity == "high")
    n_med  = sum(1 for f in report.findings if f.severity == "medium")
    n_low  = sum(1 for f in report.findings if f.severity == "low")

    card = lambda label, value, color="#f7fafc": (
        f'<div style="background:#1a202c;border:1px solid #2d3748;border-radius:8px;'
        f'padding:12px 20px;min-width:130px;flex:1">'
        f'<div style="color:#718096;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em">{label}</div>'
        f'<div style="color:{color};font-size:1.2rem;font-weight:700;margin-top:4px">{value}</div></div>'
    )

    return (
        '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:4px">'
        + card("Run ID", report.trace.run_id)
        + card("Model", report.trace.model[:28])
        + card("Turns", len(report.trace.turns))
        + card("Total Tokens", f"{report.total_tokens_billed:,}", "#fbd38d")
        + card("Total Cost", f"${report.total_cost_usd:.4f}", "#fbd38d")
        + card("Recoverable Waste", f"${report.recoverable_cost_usd:.4f} ({rec_pct:.1f}%)", "#fc8181")
        + card("Findings", f"{n_high}H / {n_med}M / {n_low}L", "#68d391")
        + "</div>"
    )


def _region_df(report: Report) -> pd.DataFrame:
    rows = []
    for rs in report.region_summaries:
        filled = round(rs.fraction * 20)
        bar = "#" * filled + "." * (20 - filled)
        rows.append({
            "Region": rs.region.value,
            "Tokens": f"{rs.total_tokens:,}",
            "Cost (USD)": f"${rs.total_cost_usd:.5f}",
            "Share": f"{rs.fraction * 100:.1f}%",
            "Visual": bar,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Region", "Tokens", "Cost (USD)", "Share", "Visual"]
    )


def _findings_df(report: Report) -> pd.DataFrame:
    rows = []
    for i, f in enumerate(report.findings_by_severity(), 1):
        rows.append({
            "#": i,
            "Type": f.kind.value,
            "Severity": f.severity.upper(),
            "Wasted Tokens": f"{f.wasted_tokens:,}",
            "Wasted Cost": f"${f.wasted_cost_usd:.5f}",
            "Turns": f"{f.first_seen_turn}–{f.last_seen_turn}",
            "Description": f.description[:100],
            "Fix": f.fix[:90],
        })
    cols = ["#", "Type", "Severity", "Wasted Tokens", "Wasted Cost", "Turns", "Description", "Fix"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


def _rebilling_df(report: Report) -> pd.DataFrame:
    rows = []
    for e in report.rebilling_entries[:20]:
        rows.append({
            "Region": e.region.value,
            "Preview": e.content_preview[:70],
            "Tokens/Turn": f"{e.token_count:,}",
            "Turns": e.turns_present,
            "Cumul. Tokens": f"{e.cumulative_tokens:,}",
            "Cumul. Cost": f"${e.cumulative_cost_usd:.5f}",
        })
    cols = ["Region", "Preview", "Tokens/Turn", "Turns", "Cumul. Tokens", "Cumul. Cost"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


def _html_file(report: Report) -> str:
    html = render_html_report(report)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8", prefix="contextlens_"
    )
    tmp.write(html)
    tmp.close()
    return tmp.name


def _run_analysis(trace_dict: dict) -> tuple:
    trace = _dict_to_trace(trace_dict)
    report = analyze_trace(trace)
    return (
        _summary_html(report),
        _region_df(report),
        _rebilling_df(report),
        _findings_df(report),
        _html_file(report),
    )


_EMPTY = pd.DataFrame()


# ---------------------------------------------------------------------------
# Gradio action functions
# ---------------------------------------------------------------------------


def run_demo() -> tuple:
    if not _IMPORT_OK:
        err_html = f"<pre style='color:#fc8181'>Import error:\n{_IMPORT_ERROR}</pre>"
        return err_html, _EMPTY, _EMPTY, _EMPTY, None
    try:
        trace_dict = _build_demo_trace()
        return _run_analysis(trace_dict)
    except Exception:
        err_html = f"<pre style='color:#fc8181'>{traceback.format_exc()}</pre>"
        return err_html, _EMPTY, _EMPTY, _EMPTY, None


def analyze_upload(file) -> tuple:
    if not _IMPORT_OK:
        err_html = f"<pre style='color:#fc8181'>Import error:\n{_IMPORT_ERROR}</pre>"
        return err_html, _EMPTY, _EMPTY, _EMPTY, None
    if file is None:
        return (
            "<p style='color:#fc8181'>Please upload a trace JSON file first.</p>",
            _EMPTY, _EMPTY, _EMPTY, None,
        )
    try:
        path = file.name if hasattr(file, "name") else str(file)
        with open(path, encoding="utf-8") as f:
            trace_dict = json.load(f)
        return _run_analysis(trace_dict)
    except Exception:
        err_html = f"<pre style='color:#fc8181'>{traceback.format_exc()}</pre>"
        return err_html, _EMPTY, _EMPTY, _EMPTY, None


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

TRACE_FORMAT_HELP = """
### Trace JSON format

Capture automatically from your agent:
```python
pip install contextlens

import contextlens as cl

# Anthropic
with cl.capture_anthropic(client) as col:
    # your agent loop
col.save("trace.json")

# OpenAI
with cl.capture_openai(client) as col:
    # your agent loop
col.save("trace.json")
```

Or build a trace dict manually — each turn needs:
`turn_index`, `model`, `provider`, `tool_names_defined`,
`tool_names_called`, `raw_request` (with `messages`, `system`, `tools`).
"""

# Gradio 5/6 compatible theme setup
_gradio_major = int(gr.__version__.split(".")[0])
_theme = gr.themes.Base(primary_hue="orange", neutral_hue="slate")
_blocks_kwargs: dict = {"title": "ContextLens — LLM Context Profiler"}
if _gradio_major < 6:
    _blocks_kwargs["theme"] = _theme

with gr.Blocks(**_blocks_kwargs) as demo:

    gr.Markdown(
        """
# ContextLens 🔬
### Diagnostic profiler for LLM agent context windows — *py-spy / pprof, but for what's inside your prompt*

In multi-turn agent loops **the full context is re-sent on every API call**.
ContextLens decomposes the window, tracks re-billing cost per block across turns,
and surfaces waste as ranked findings with dollar costs and one-line fixes.

**Five waste detectors:** Duplicate · Near-duplicate (Jaccard ≥ 0.85) · Stale tool results · Unused tool schemas · Redundant retrieval chunks

> **No API key needed. No file required. Click Run Demo to see live results.**

[![GitHub](https://img.shields.io/badge/GitHub-HarshalSant%2Fcontextlens-black?logo=github)](https://github.com/HarshalSant/contextlens)
&nbsp;&nbsp;`pip install contextlens-profiler`
---
"""
    )

    with gr.Tabs():

        # ── Tab 1: Live demo ──────────────────────────────────────────────
        with gr.Tab("🚀 Live Demo  (no upload needed)"):
            gr.Markdown(
                "Simulates a **30-turn JWT-migration agent loop** with canned data. "
                "Fires all five waste detectors. **Click the button below** — no API key, no file, no signup needed. "
                "Results appear in ~2 seconds."
            )
            demo_btn = gr.Button("▶  Run Demo Analysis", variant="primary", size="lg")
            demo_summary = gr.HTML(label="Run Summary")
            with gr.Tabs():
                with gr.Tab("Region Breakdown"):
                    demo_region = gr.Dataframe(interactive=False, wrap=True)
                with gr.Tab("Re-Billing Top 20"):
                    demo_rebill = gr.Dataframe(interactive=False, wrap=True)
                with gr.Tab("Waste Findings"):
                    demo_findings = gr.Dataframe(interactive=False, wrap=True)
            demo_file = gr.File(label="Download Interactive HTML Report (.html)", file_types=[".html"])

            demo_btn.click(
                fn=run_demo,
                inputs=[],
                outputs=[demo_summary, demo_region, demo_rebill, demo_findings, demo_file],
                api_name=False,
            )

        # ── Tab 2: Upload trace ───────────────────────────────────────────
        with gr.Tab("📂 Analyze Your Trace"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload = gr.File(label="Upload trace.json", file_types=[".json"])
                    analyze_btn = gr.Button("Analyze", variant="primary")
                    gr.Markdown(TRACE_FORMAT_HELP)
                with gr.Column(scale=2):
                    up_summary  = gr.HTML(label="Run Summary")
                    with gr.Tabs():
                        with gr.Tab("Region Breakdown"):
                            up_region = gr.Dataframe(interactive=False, wrap=True)
                        with gr.Tab("Re-Billing Top 20"):
                            up_rebill = gr.Dataframe(interactive=False, wrap=True)
                        with gr.Tab("Waste Findings"):
                            up_findings = gr.Dataframe(interactive=False, wrap=True)
                    up_file = gr.File(label="Download Interactive HTML Report", file_types=[".html"])

            analyze_btn.click(
                fn=analyze_upload,
                inputs=[upload],
                outputs=[up_summary, up_region, up_rebill, up_findings, up_file],
                api_name=False,
            )

    gr.Markdown(
        "---\n"
        "MIT License · [GitHub](https://github.com/HarshalSant/contextlens) · "
        "`pip install contextlens-profiler` · Built with Python 3.11+"
    )


if __name__ == "__main__":
    _launch_kwargs: dict = {}
    if _gradio_major >= 6:
        _launch_kwargs["theme"] = _theme
    demo.launch(**_launch_kwargs)
