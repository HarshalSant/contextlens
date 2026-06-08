"""CLI entry-point for ContextLens."""

from __future__ import annotations

import sys

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .analyzer import analyze_file
from .models import Report, WasteKind
from .reporter import render_html_report

console = Console()

SEVERITY_COLORS = {
    "high": "bold red",
    "medium": "yellow",
    "low": "dim",
}

KIND_ICONS = {
    WasteKind.DUPLICATE: "[D]",
    WasteKind.NEAR_DUPLICATE: "[~D]",
    WasteKind.STALE_TOOL_RESULT: "[S]",
    WasteKind.UNUSED_TOOL_SCHEMA: "[U]",
    WasteKind.REDUNDANT_RETRIEVAL: "[R]",
}


@click.group()
@click.version_option(package_name="contextlens")
def main() -> None:
    """ContextLens — diagnostic profiler for LLM agent context windows."""


@main.command("analyze")
@click.argument("trace_file", type=click.Path(exists=True))
@click.option("--model", default=None, help="Override the model name for cost calculation.")
@click.option("--top", default=10, show_default=True, help="Number of waste findings to show.")
def cmd_analyze(trace_file: str, model: str | None, top: int) -> None:
    """Print a ranked waste report for TRACE_FILE to the terminal."""
    try:
        report = analyze_file(trace_file)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    _print_report(report, top=top)


@main.command("report")
@click.argument("trace_file", type=click.Path(exists=True))
@click.option("-o", "--output", default="report.html", show_default=True, help="Output HTML path.")
def cmd_report(trace_file: str, output: str) -> None:
    """Generate an interactive HTML treemap report for TRACE_FILE."""
    try:
        report = analyze_file(trace_file)
        html = render_html_report(report)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    console.print(f"[bold green]Report written to:[/bold green] {output}")
    _print_summary_line(report)


def _print_report(report: Report, top: int = 10) -> None:
    """Print a full terminal report."""
    trace = report.trace
    n_turns = len(trace.turns)

    # Header
    console.print(
        Panel(
            f"[bold]ContextLens[/bold] | Run [cyan]{trace.run_id}[/cyan]\n"
            f"Model: [yellow]{trace.model}[/yellow]  |  "
            f"Provider: [yellow]{trace.provider}[/yellow]  |  "
            f"Turns: [cyan]{n_turns}[/cyan]",
            box=box.ROUNDED,
            expand=False,
        )
    )

    # Token composition by region
    region_table = Table(
        title="Context Composition by Region",
        box=box.SIMPLE_HEAVY,
        show_footer=True,
    )
    region_table.add_column("Region", style="bold")
    region_table.add_column("Tokens", justify="right")
    region_table.add_column("Cost (USD)", justify="right")
    region_table.add_column("Share", justify="right")

    total_tok = report.total_tokens_billed
    total_cost = report.total_cost_usd

    for rs in report.region_summaries:
        pct = f"{rs.fraction * 100:.1f}%"
        region_table.add_row(
            rs.region.value,
            f"{rs.total_tokens:,}",
            f"${rs.total_cost_usd:.4f}",
            _bar(rs.fraction) + f" {pct}",
        )

    region_table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_tok:,}[/bold]",
        f"[bold]${total_cost:.4f}[/bold]",
        "",
        end_section=True,
    )
    console.print(region_table)

    # Re-billing summary
    rec_pct = (
        report.recoverable_tokens / report.total_tokens_billed * 100
        if report.total_tokens_billed
        else 0.0
    )
    console.print(
        f"\n[bold]Re-billing:[/bold] {report.total_tokens_billed:,} tokens billed across "
        f"{n_turns} turns  ->  "
        f"[yellow]{report.recoverable_tokens:,}[/yellow] tokens ({rec_pct:.1f}%) are re-billing waste  "
        f"[red](${report.recoverable_cost_usd:.4f})[/red]\n"
    )

    # Findings table
    findings = report.findings_by_severity()[:top]
    if not findings:
        console.print("[green]No waste findings detected.[/green]")
        return

    find_table = Table(
        title=f"Top {min(top, len(findings))} Waste Findings",
        box=box.SIMPLE_HEAVY,
    )
    find_table.add_column("#", justify="right", style="dim")
    find_table.add_column("Type")
    find_table.add_column("Sev")
    find_table.add_column("Wasted Tokens", justify="right")
    find_table.add_column("Cost (USD)", justify="right")
    find_table.add_column("Description")
    find_table.add_column("Fix")

    for idx, finding in enumerate(findings, 1):
        sev_color = SEVERITY_COLORS.get(finding.severity, "")
        icon = KIND_ICONS.get(finding.kind, "")
        find_table.add_row(
            str(idx),
            f"{icon} {finding.kind.value}",
            f"[{sev_color}]{finding.severity}[/{sev_color}]",
            f"{finding.wasted_tokens:,}",
            f"${finding.wasted_cost_usd:.4f}",
            finding.description[:60] + ("…" if len(finding.description) > 60 else ""),
            finding.fix[:60] + ("…" if len(finding.fix) > 60 else ""),
        )

    console.print(find_table)

    total_waste_tokens = sum(f.wasted_tokens for f in report.findings)
    total_waste_cost = sum(f.wasted_cost_usd for f in report.findings)
    console.print(
        f"\n[bold red]Total addressable waste:[/bold red] "
        f"{total_waste_tokens:,} tokens  /  ${total_waste_cost:.4f}\n"
    )


def _print_summary_line(report: Report) -> None:
    n = len(report.findings)
    console.print(
        f"[dim]{n} finding(s), {report.total_tokens_billed:,} tokens billed, "
        f"${report.total_cost_usd:.4f} total, "
        f"${report.recoverable_cost_usd:.4f} recoverable[/dim]"
    )


def _bar(fraction: float, width: int = 10) -> str:
    filled = round(fraction * width)
    return "#" * filled + "." * (width - filled)
