"""ContextLens — diagnostic profiler for LLM agent context windows."""

from .analyzer import analyze_file, analyze_trace
from .capture import (
    TraceCollector,
    capture_anthropic,
    capture_openai,
    load_trace,
    load_trace_from_dict,
)
from .costs import CostModel, ModelPricing, default_cost_model
from .models import (
    ContentBlock,
    Finding,
    RebillingEntry,
    Region,
    RegionSummary,
    Report,
    Trace,
    TurnSnapshot,
    WasteKind,
)
from .reporter import render_html_report

__version__ = "0.1.0"
__all__ = [
    # Analysis
    "analyze_trace",
    "analyze_file",
    # Capture
    "TraceCollector",
    "capture_anthropic",
    "capture_openai",
    "load_trace",
    "load_trace_from_dict",
    # Models
    "Trace",
    "TurnSnapshot",
    "ContentBlock",
    "Region",
    "WasteKind",
    "Finding",
    "RebillingEntry",
    "RegionSummary",
    "Report",
    # Costs
    "CostModel",
    "ModelPricing",
    "default_cost_model",
    # Report
    "render_html_report",
]
