"""Generate a self-contained, single-file HTML report with a D3 treemap."""

from __future__ import annotations

import json
from typing import Any

from .models import Region, Report, WasteKind

# Color palette for regions
REGION_COLORS: dict[str, str] = {
    Region.SYSTEM.value: "#4e79a7",
    Region.TOOL_SCHEMA.value: "#f28e2b",
    Region.TOOL_RESULT.value: "#e15759",
    Region.USER_MESSAGE.value: "#76b7b2",
    Region.ASSISTANT_MESSAGE.value: "#59a14f",
    Region.RETRIEVED_CONTENT.value: "#edc948",
    Region.UNKNOWN.value: "#b07aa1",
}

WASTE_KIND_LABELS: dict[str, str] = {
    WasteKind.DUPLICATE.value: "Duplicate",
    WasteKind.NEAR_DUPLICATE.value: "Near-Duplicate",
    WasteKind.STALE_TOOL_RESULT.value: "Stale Tool Result",
    WasteKind.UNUSED_TOOL_SCHEMA.value: "Unused Tool Schema",
    WasteKind.REDUNDANT_RETRIEVAL.value: "Redundant Retrieval",
}


def render_html_report(report: Report) -> str:
    """Return a complete, self-contained HTML page string."""
    data = _build_report_data(report)
    data_json = json.dumps(data, indent=2)
    return _HTML_TEMPLATE.replace("__REPORT_DATA__", data_json)


def _build_report_data(report: Report) -> dict[str, Any]:
    trace = report.trace

    # Build treemap hierarchy: root → region → block (per turn)
    region_children: dict[str, list[dict[str, Any]]] = {}

    for rs in report.region_summaries:
        region_name = rs.region.value
        region_children[region_name] = []

    for turn in trace.turns:
        for block in turn.blocks:
            region_name = block.region.value
            if region_name not in region_children:
                region_children[region_name] = []
            region_children[region_name].append(
                {
                    "name": f"Turn {block.turn_index} • {block.region.value}",
                    "value": block.token_count,
                    "turn": block.turn_index,
                    "preview": block.content[:120].replace('"', "'"),
                    "region": region_name,
                    "tool": block.tool_name or "",
                }
            )

    treemap_root = {
        "name": "Context Window",
        "children": [
            {
                "name": region_name,
                "color": REGION_COLORS.get(region_name, "#aaa"),
                "children": children,
            }
            for region_name, children in region_children.items()
            if children
        ],
    }

    # Timeline: tokens per region per turn
    turn_labels = [f"T{t.turn_index}" for t in trace.turns]
    region_series: dict[str, list[int]] = {r.value: [] for r in Region}
    for turn in trace.turns:
        by_region: dict[str, int] = {r.value: 0 for r in Region}
        for block in turn.blocks:
            by_region[block.region.value] += block.token_count
        for region_val in region_series:
            region_series[region_val].append(by_region[region_val])

    # Findings
    findings_data = []
    for f in report.findings_by_severity():
        findings_data.append(
            {
                "kind": WASTE_KIND_LABELS.get(f.kind.value, f.kind.value),
                "severity": f.severity,
                "description": f.description,
                "fix": f.fix,
                "wasted_tokens": f.wasted_tokens,
                "wasted_cost_usd": round(f.wasted_cost_usd, 6),
            }
        )

    return {
        "meta": {
            "run_id": trace.run_id,
            "model": trace.model,
            "provider": trace.provider,
            "total_turns": len(trace.turns),
            "total_tokens_billed": report.total_tokens_billed,
            "total_cost_usd": round(report.total_cost_usd, 6),
            "recoverable_tokens": report.recoverable_tokens,
            "recoverable_cost_usd": round(report.recoverable_cost_usd, 6),
        },
        "region_summaries": [
            {
                "region": rs.region.value,
                "total_tokens": rs.total_tokens,
                "total_cost_usd": round(rs.total_cost_usd, 6),
                "fraction": round(rs.fraction, 4),
                "color": REGION_COLORS.get(rs.region.value, "#aaa"),
            }
            for rs in report.region_summaries
        ],
        "treemap": treemap_root,
        "timeline": {
            "turns": turn_labels,
            "series": [
                {
                    "region": region_val,
                    "color": REGION_COLORS.get(region_val, "#aaa"),
                    "data": vals,
                }
                for region_val, vals in region_series.items()
                if any(v > 0 for v in vals)
            ],
        },
        "findings": findings_data,
        "rebilling_top20": [
            {
                "preview": e.content_preview,
                "region": e.region.value,
                "token_count": e.token_count,
                "turns_present": e.turns_present,
                "cumulative_tokens": e.cumulative_tokens,
                "cumulative_cost_usd": round(e.cumulative_cost_usd, 6),
            }
            for e in report.rebilling_entries[:20]
        ],
    }


# ---------------------------------------------------------------------------
# Self-contained HTML template with inlined D3 v7 (CDN, but cached)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ContextLens - Context Window Diagnostic Report</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: #0f1117; color: #e2e8f0; font-size: 14px; }
    .container { max-width: 1400px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 1.6rem; font-weight: 700; color: #f7fafc; }
    h2 { font-size: 1.1rem; font-weight: 600; margin: 24px 0 12px; color: #a0aec0; text-transform: uppercase; letter-spacing: .05em; }
    .meta-bar { display: flex; gap: 24px; flex-wrap: wrap; margin: 16px 0; }
    .stat { background: #1a202c; border: 1px solid #2d3748; border-radius: 8px; padding: 12px 20px; min-width: 140px; }
    .stat-label { font-size: .75rem; color: #718096; text-transform: uppercase; letter-spacing: .05em; }
    .stat-value { font-size: 1.5rem; font-weight: 700; color: #f7fafc; margin-top: 4px; }
    .stat-value.danger { color: #fc8181; }
    .stat-value.warn  { color: #fbd38d; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
    .card { background: #1a202c; border: 1px solid #2d3748; border-radius: 12px; padding: 20px; }
    #treemap { width: 100%; height: 420px; }
    #timeline { width: 100%; height: 240px; }
    .tooltip { position: absolute; pointer-events: none; background: #2d3748;
               border: 1px solid #4a5568; border-radius: 6px; padding: 10px 14px;
               font-size: 12px; color: #e2e8f0; max-width: 320px; z-index: 100; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #2d3748; }
    th { color: #718096; font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; background: #171923; }
    tr:hover td { background: #2d3748; }
    .badge { display: inline-block; border-radius: 4px; padding: 2px 8px; font-size: .7rem; font-weight: 600; text-transform: uppercase; }
    .badge-high   { background: #742a2a; color: #fc8181; }
    .badge-medium { background: #744210; color: #fbd38d; }
    .badge-low    { background: #1a202c; color: #718096; }
    .legend { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
    .legend-item { display: flex; align-items: center; gap: 6px; font-size: .8rem; }
    .legend-dot { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }
    .fix-text { color: #68d391; font-size: .8rem; }
    .section { margin-bottom: 32px; }
  </style>
</head>
<body>
<div class="container">
  <h1>ContextLens <span style="color:#718096;font-weight:400;font-size:1rem;">— Context Window Diagnostic Report</span></h1>

  <div id="meta-bar" class="meta-bar"></div>

  <div class="section">
    <h2>Context Composition Treemap</h2>
    <div class="legend" id="treemap-legend"></div>
    <div class="card"><div id="treemap"></div></div>
  </div>

  <div class="section">
    <h2>Token Usage Over Turns</h2>
    <div class="card"><svg id="timeline"></svg></div>
  </div>

  <div class="grid2 section">
    <div class="card">
      <h2 style="margin-top:0">Region Breakdown</h2>
      <table id="region-table">
        <thead><tr><th>Region</th><th>Tokens</th><th>Cost (USD)</th><th>Share</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="card">
      <h2 style="margin-top:0">Top Re-Billed Blocks</h2>
      <table id="rebill-table">
        <thead><tr><th>Preview</th><th>Region</th><th>Toks</th><th>Turns</th><th>Cumul. Cost</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <h2>Waste Findings</h2>
    <table id="findings-table">
      <thead>
        <tr><th>#</th><th>Type</th><th>Sev.</th><th>Wasted Tokens</th><th>Cost (USD)</th><th>Description</th><th>Fix</th></tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<div class="tooltip" id="tooltip" style="display:none"></div>

<script>
const DATA = __REPORT_DATA__;

// ---- Meta bar ----
const meta = DATA.meta;
const metaBar = document.getElementById("meta-bar");
const stats = [
  { label: "Run ID",       value: meta.run_id },
  { label: "Model",        value: meta.model },
  { label: "Turns",        value: meta.total_turns },
  { label: "Total Tokens", value: meta.total_tokens_billed.toLocaleString(), cls: "" },
  { label: "Total Cost",   value: "$" + meta.total_cost_usd.toFixed(4), cls: "warn" },
  { label: "Recoverable",  value: "$" + meta.recoverable_cost_usd.toFixed(4), cls: "danger" },
];
stats.forEach(s => {
  metaBar.innerHTML += `<div class="stat"><div class="stat-label">${s.label}</div>
    <div class="stat-value ${s.cls||''}">${s.value}</div></div>`;
});

// ---- Tooltip helper ----
const tooltip = document.getElementById("tooltip");
function showTip(html, x, y) {
  tooltip.innerHTML = html;
  tooltip.style.display = "block";
  tooltip.style.left = (x + 14) + "px";
  tooltip.style.top  = (y - 10) + "px";
}
function hideTip() { tooltip.style.display = "none"; }

// ---- Treemap legend ----
const legend = document.getElementById("treemap-legend");
DATA.region_summaries.forEach(rs => {
  legend.innerHTML += `<div class="legend-item">
    <div class="legend-dot" style="background:${rs.color}"></div>
    <span>${rs.region} (${(rs.fraction*100).toFixed(1)}%)</span></div>`;
});

// ---- D3 Treemap ----
(function() {
  const el = document.getElementById("treemap");
  const W = el.clientWidth || 900, H = 420;
  const svg = d3.select(el).append("svg").attr("width", W).attr("height", H);

  const root = d3.hierarchy(DATA.treemap)
    .sum(d => d.value || 0)
    .sort((a, b) => (b.value || 0) - (a.value || 0));

  d3.treemap().size([W, H]).paddingOuter(4).paddingInner(2).round(true)(root);

  // Color: leaves get their region color, parent = white outline
  function leafColor(d) {
    const region = d.data.region || (d.parent && d.parent.data.name);
    const rs = DATA.region_summaries.find(r => r.region === region);
    return rs ? rs.color : "#4a5568";
  }

  const leaves = root.leaves();
  const g = svg.selectAll("g").data(leaves).enter().append("g")
    .attr("transform", d => `translate(${d.x0},${d.y0})`);

  g.append("rect")
    .attr("width",  d => Math.max(0, d.x1 - d.x0))
    .attr("height", d => Math.max(0, d.y1 - d.y0))
    .attr("fill",   d => leafColor(d))
    .attr("opacity", 0.82)
    .attr("stroke", "#0f1117")
    .attr("stroke-width", 1)
    .style("cursor", "pointer")
    .on("mousemove", function(event, d) {
      const costEst = (d.value / meta.total_tokens_billed * meta.total_cost_usd).toFixed(6);
      showTip(
        `<strong>${d.data.region || d.parent.data.name}</strong><br>
         Turn: ${d.data.turn !== undefined ? d.data.turn : "—"}<br>
         Tokens: ${(d.value||0).toLocaleString()}<br>
         Est. Cost: $${costEst}<br>
         ${d.data.tool ? "Tool: " + d.data.tool + "<br>" : ""}
         <em style="color:#a0aec0">${(d.data.preview||"").slice(0,100)}</em>`,
        event.pageX, event.pageY
      );
    })
    .on("mouseleave", hideTip);

  // Labels (only if rectangle is large enough)
  g.filter(d => (d.x1 - d.x0) > 40 && (d.y1 - d.y0) > 18)
    .append("text")
    .attr("x", 4).attr("y", 14)
    .attr("fill", "#fff").attr("font-size", 11)
    .attr("pointer-events", "none")
    .text(d => d.data.region || "");
})();

// ---- Timeline stacked area chart ----
(function() {
  const tl = DATA.timeline;
  if (!tl.turns.length) return;

  const margin = { top: 10, right: 20, bottom: 30, left: 60 };
  const svgEl = document.getElementById("timeline");
  const W = svgEl.parentElement.clientWidth || 900;
  const H = 240;
  svgEl.setAttribute("width", W);
  svgEl.setAttribute("height", H);
  const width  = W - margin.left - margin.right;
  const height = H - margin.top  - margin.bottom;

  const svg = d3.select(svgEl).append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const series = tl.series;
  const n = tl.turns.length;

  // Build stacked data
  const stackKeys = series.map(s => s.region);
  const rows = tl.turns.map((t, i) => {
    const row = { turn: t };
    series.forEach(s => { row[s.region] = s.data[i] || 0; });
    return row;
  });

  const stack = d3.stack().keys(stackKeys).order(d3.stackOrderNone).offset(d3.stackOffsetNone);
  const stacked = stack(rows);

  const xScale = d3.scaleBand().domain(tl.turns).range([0, width]).padding(0.05);
  const maxY = d3.max(stacked, layer => d3.max(layer, d => d[1])) || 1;
  const yScale = d3.scaleLinear().domain([0, maxY]).range([height, 0]);

  const colorMap = {};
  series.forEach(s => { colorMap[s.region] = s.color; });

  // Stacked bars
  svg.selectAll(".layer")
    .data(stacked)
    .enter().append("g")
      .attr("class", "layer")
      .attr("fill", d => colorMap[d.key] || "#4a5568")
    .selectAll("rect")
    .data(d => d)
    .enter().append("rect")
      .attr("x",      d => xScale(d.data.turn))
      .attr("y",      d => yScale(d[1]))
      .attr("height", d => yScale(d[0]) - yScale(d[1]))
      .attr("width",  xScale.bandwidth());

  // Axes
  const xAxis = d3.axisBottom(xScale).tickValues(
    tl.turns.filter((_, i) => i % Math.max(1, Math.floor(n / 15)) === 0)
  );
  svg.append("g").attr("transform", `translate(0,${height})`)
    .call(xAxis)
    .selectAll("text").attr("fill", "#718096").attr("font-size", 10);
  svg.append("g").call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format(".2s")))
    .selectAll("text").attr("fill", "#718096").attr("font-size", 10);
  svg.selectAll(".domain,.tick line").attr("stroke", "#2d3748");
})();

// ---- Region table ----
(function() {
  const tbody = document.querySelector("#region-table tbody");
  DATA.region_summaries.forEach(rs => {
    const pct = (rs.fraction * 100).toFixed(1);
    const bar = "█".repeat(Math.round(rs.fraction * 20)) + "░".repeat(20 - Math.round(rs.fraction * 20));
    tbody.innerHTML += `<tr>
      <td><span style="display:inline-block;width:10px;height:10px;background:${rs.color};border-radius:2px;margin-right:6px"></span>${rs.region}</td>
      <td>${rs.total_tokens.toLocaleString()}</td>
      <td>$${rs.total_cost_usd.toFixed(4)}</td>
      <td style="font-family:monospace;color:#4a5568">${bar} ${pct}%</td>
    </tr>`;
  });
})();

// ---- Rebilling table ----
(function() {
  const tbody = document.querySelector("#rebill-table tbody");
  DATA.rebilling_top20.slice(0,15).forEach(e => {
    tbody.innerHTML += `<tr>
      <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${e.preview}">${e.preview}</td>
      <td>${e.region}</td>
      <td>${e.token_count.toLocaleString()}</td>
      <td>${e.turns_present}</td>
      <td style="color:#fc8181">$${e.cumulative_cost_usd.toFixed(4)}</td>
    </tr>`;
  });
})();

// ---- Findings table ----
(function() {
  const tbody = document.querySelector("#findings-table tbody");
  DATA.findings.forEach((f, i) => {
    tbody.innerHTML += `<tr>
      <td style="color:#718096">${i+1}</td>
      <td>${f.kind}</td>
      <td><span class="badge badge-${f.severity}">${f.severity}</span></td>
      <td>${f.wasted_tokens.toLocaleString()}</td>
      <td style="color:#fc8181">$${f.wasted_cost_usd.toFixed(4)}</td>
      <td style="max-width:260px">${f.description}</td>
      <td class="fix-text" style="max-width:220px">${f.fix}</td>
    </tr>`;
  });
  if (!DATA.findings.length) {
    tbody.innerHTML = '<tr><td colspan="7" style="color:#68d391;text-align:center">No waste findings detected.</td></tr>';
  }
})();
</script>
</body>
</html>
"""
