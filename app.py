"""
ContextLens — Hugging Face Spaces Gradio app.

Two entry points:
  Tab 1: Run the built-in 30-turn demo (no upload needed).
  Tab 2: Upload your own trace JSON and get a full analysis.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Allow import when running directly without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import pandas as pd

from contextlens.analyzer import analyze_trace
from contextlens.capture import _dict_to_trace
from contextlens.models import Report
from contextlens.reporter import render_html_report

# ---------------------------------------------------------------------------
# Reuse the demo trace builder from examples/demo.py
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent / "examples"))
from demo import build_demo_trace  # type: ignore[import]


# ---------------------------------------------------------------------------
# Analysis helpers — turn a Report into Gradio-friendly outputs
# ---------------------------------------------------------------------------


def _summary_html(report: Report) -> str:
    meta = report.trace
    rec_pct = (
        report.recoverable_tokens / report.total_tokens_billed * 100
        if report.total_tokens_billed
        else 0.0
    )
    return f"""
<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px">
  <div style="background:#1a202c;border:1px solid #2d3748;border-radius:8px;padding:12px 20px;min-width:130px">
    <div style="color:#718096;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em">Run ID</div>
    <div style="color:#f7fafc;font-size:1.1rem;font-weight:700">{meta.run_id}</div>
  </div>
  <div style="background:#1a202c;border:1px solid #2d3748;border-radius:8px;padding:12px 20px;min-width:130px">
    <div style="color:#718096;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em">Model</div>
    <div style="color:#f7fafc;font-size:1.1rem;font-weight:700">{meta.model}</div>
  </div>
  <div style="background:#1a202c;border:1px solid #2d3748;border-radius:8px;padding:12px 20px;min-width:130px">
    <div style="color:#718096;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em">Turns</div>
    <div style="color:#f7fafc;font-size:1.1rem;font-weight:700">{len(meta.turns)}</div>
  </div>
  <div style="background:#1a202c;border:1px solid #2d3748;border-radius:8px;padding:12px 20px;min-width:130px">
    <div style="color:#718096;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em">Total Tokens</div>
    <div style="color:#fbd38d;font-size:1.1rem;font-weight:700">{report.total_tokens_billed:,}</div>
  </div>
  <div style="background:#1a202c;border:1px solid #2d3748;border-radius:8px;padding:12px 20px;min-width:130px">
    <div style="color:#718096;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em">Total Cost</div>
    <div style="color:#fbd38d;font-size:1.1rem;font-weight:700">${report.total_cost_usd:.4f}</div>
  </div>
  <div style="background:#742a2a;border:1px solid #fc8181;border-radius:8px;padding:12px 20px;min-width:130px">
    <div style="color:#fc8181;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em">Recoverable Waste</div>
    <div style="color:#fc8181;font-size:1.1rem;font-weight:700">${report.recoverable_cost_usd:.4f} ({rec_pct:.1f}%)</div>
  </div>
  <div style="background:#1a202c;border:1px solid #2d3748;border-radius:8px;padding:12px 20px;min-width:130px">
    <div style="color:#718096;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em">Findings</div>
    <div style="color:#68d391;font-size:1.1rem;font-weight:700">{len(report.findings)}</div>
  </div>
</div>
"""


def _region_df(report: Report) -> pd.DataFrame:
    rows = []
    for rs in report.region_summaries:
        bar = "#" * round(rs.fraction * 20) + "." * (20 - round(rs.fraction * 20))
        rows.append(
            {
                "Region": rs.region.value,
                "Tokens": f"{rs.total_tokens:,}",
                "Cost (USD)": f"${rs.total_cost_usd:.5f}",
                "Share": f"{rs.fraction * 100:.1f}%",
                "Visual": bar,
            }
        )
    return pd.DataFrame(rows)


def _findings_df(report: Report) -> pd.DataFrame:
    rows = []
    for i, f in enumerate(report.findings_by_severity(), 1):
        rows.append(
            {
                "#": i,
                "Type": f.kind.value,
                "Severity": f.severity.upper(),
                "Wasted Tokens": f"{f.wasted_tokens:,}",
                "Wasted Cost": f"${f.wasted_cost_usd:.5f}",
                "Turns": f"{f.first_seen_turn}–{f.last_seen_turn}",
                "Description": f.description[:120],
                "Fix": f.fix[:100],
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["#", "Type", "Severity", "Wasted Tokens", "Wasted Cost", "Turns", "Description", "Fix"]
        )
    return pd.DataFrame(rows)


def _rebilling_df(report: Report) -> pd.DataFrame:
    rows = []
    for e in report.rebilling_entries[:20]:
        rows.append(
            {
                "Region": e.region.value,
                "Preview": e.content_preview[:80],
                "Tokens/Turn": f"{e.token_count:,}",
                "Turns Present": e.turns_present,
                "Cumul. Tokens": f"{e.cumulative_tokens:,}",
                "Cumul. Cost": f"${e.cumulative_cost_usd:.5f}",
            }
        )
    return pd.DataFrame(rows)


def _html_report_file(report: Report) -> str:
    """Write HTML report to a temp file and return the path."""
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
    summary = _summary_html(report)
    region_df = _region_df(report)
    findings_df = _findings_df(report)
    rebilling_df = _rebilling_df(report)
    html_path = _html_report_file(report)
    return summary, region_df, findings_df, rebilling_df, html_path


# ---------------------------------------------------------------------------
# Tab actions
# ---------------------------------------------------------------------------


def run_demo() -> tuple:
    """Run the built-in 30-turn canned demo."""
    trace_dict = build_demo_trace()
    return _run_analysis(trace_dict)


def analyze_upload(file) -> tuple:
    """Analyze a user-uploaded trace JSON file."""
    if file is None:
        empty = pd.DataFrame()
        return (
            "<p style='color:#fc8181'>Please upload a trace JSON file.</p>",
            empty, empty, empty, None,
        )
    try:
        with open(file.name, encoding="utf-8") as f:
            trace_dict = json.load(f)
        return _run_analysis(trace_dict)
    except Exception as e:
        empty = pd.DataFrame()
        return (
            f"<p style='color:#fc8181'>Error parsing trace: {e}</p>",
            empty, empty, empty, None,
        )


# ---------------------------------------------------------------------------
# Result block — shared between both tabs
# ---------------------------------------------------------------------------


def _result_block() -> list:
    """Return a list of Gradio components used to display results."""
    summary_html = gr.HTML(label="Summary")
    with gr.Tabs():
        with gr.Tab("Region Breakdown"):
            region_table = gr.Dataframe(
                label="Tokens by Region",
                interactive=False,
                wrap=True,
            )
        with gr.Tab("Re-Billing (Top 20)"):
            rebilling_table = gr.Dataframe(
                label="Most Expensive Repeated Blocks",
                interactive=False,
                wrap=True,
            )
        with gr.Tab("Waste Findings"):
            findings_table = gr.Dataframe(
                label="Ranked Waste Findings",
                interactive=False,
                wrap=True,
            )
    html_file = gr.File(label="Download Interactive HTML Report", file_types=[".html"])
    return [summary_html, region_table, rebilling_table, findings_table, html_file]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
body { background: #0f1117 !important; }
.gradio-container { max-width: 1200px; margin: 0 auto; }
h1 { color: #f7fafc; }
.gr-form { background: #1a202c !important; border: 1px solid #2d3748 !important; }
footer { display: none !important; }
"""

DESCRIPTION = """
## ContextLens — Context Window Diagnostic Profiler

**py-spy / pprof, but for what's inside your prompt.**

In multi-turn agent loops every call re-sends the full context. ContextLens decomposes the window,
tracks re-billing cost over turns, and surfaces waste as ranked findings with dollar costs and concrete fixes.

**Five waste detectors:** Duplicate content · Near-duplicate (Jaccard) · Stale tool results · Unused tool schemas · Redundant retrieval chunks

---
"""

TRACE_FORMAT_HELP = """
### Trace JSON format

Your trace should be a JSON object with this structure:

```json
{
  "run_id": "my-run-001",
  "model": "claude-3-5-sonnet-20241022",
  "provider": "anthropic",
  "turns": [
    {
      "turn_index": 0,
      "model": "claude-3-5-sonnet-20241022",
      "provider": "anthropic",
      "total_tokens": 1200,
      "tool_names_defined": ["search_code"],
      "tool_names_called": ["search_code"],
      "raw_request": {
        "system": "You are helpful.",
        "tools": [...],
        "messages": [...]
      }
    }
  ]
}
```

**Capture automatically** using the Python SDK:
```python
import contextlens as cl
with cl.capture_anthropic(client) as collector:
    # ... your agent loop ...
collector.save("trace.json")
```
"""

with gr.Blocks(css=CSS, title="ContextLens — LLM Context Profiler") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():

        # ---- Tab 1: Built-in demo ----------------------------------------
        with gr.Tab("Live Demo (no upload needed)"):
            gr.Markdown(
                "Runs a simulated **30-turn agent loop** (JWT migration task) with canned data. "
                "Designed to trigger all five waste detectors. No API key or file needed."
            )
            demo_btn = gr.Button("Run Demo Analysis", variant="primary", size="lg")
            demo_outputs = _result_block()
            demo_btn.click(
                fn=run_demo,
                inputs=[],
                outputs=demo_outputs,
            )

        # ---- Tab 2: Upload your own trace ---------------------------------
        with gr.Tab("Analyze Your Trace"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload = gr.File(
                        label="Upload trace JSON",
                        file_types=[".json"],
                    )
                    analyze_btn = gr.Button("Analyze", variant="primary")
                    gr.Markdown(TRACE_FORMAT_HELP)
                with gr.Column(scale=2):
                    upload_outputs = _result_block()

            analyze_btn.click(
                fn=analyze_upload,
                inputs=[upload],
                outputs=upload_outputs,
            )

    gr.Markdown(
        "---\n"
        "**[GitHub](https://github.com/HarshalSant/contextlens)** | "
        "`pip install contextlens` | MIT License"
    )

if __name__ == "__main__":
    demo.launch()
