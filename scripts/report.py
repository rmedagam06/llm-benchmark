#!/usr/bin/env python
"""CLI: print optimization comparison table from DB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Print benchmark comparison table from DB")
    parser.add_argument("--model", default=None, help="Filter by model name")
    parser.add_argument("--top", type=int, default=10, help="Max rows to show")
    args = parser.parse_args()

    from rich.console import Console
    from rich.table import Table

    from src.storage.db import get_all_benchmarks, init_db

    init_db()
    console = Console()

    rows = get_all_benchmarks(model_name=args.model)
    if not rows:
        console.print("[yellow]No benchmark results found in DB.[/yellow]")
        return

    rows = rows[-args.top :]
    baseline = rows[0]

    table = Table(title="Benchmark Results", show_lines=True)
    table.add_column("ID", justify="right", style="dim")
    table.add_column("Model / Config", style="cyan", min_width=28)
    table.add_column("Accuracy", justify="right")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("VRAM (MB)", justify="right")
    table.add_column("Δ Accuracy", justify="right")
    table.add_column("Timestamp", style="dim")

    for r in rows:
        delta = (r.accuracy - baseline.accuracy) * 100
        delta_str = f"{delta:+.1f}%" if r is not baseline else "—"
        cfg = r.model_name
        if r.adapter_name:
            cfg += " + LoRA"
        if r.quant_level:
            cfg += f" [{r.quant_level}]"
        table.add_row(
            str(r.id),
            cfg,
            f"{r.accuracy:.1%}",
            f"{r.avg_ttft_ms:.0f}",
            f"{r.peak_vram_mb:.0f}",
            delta_str,
            r.timestamp[:19],
        )

    console.print(table)

    best = max(rows, key=lambda r: r.accuracy)
    acc_gain = (best.accuracy - baseline.accuracy) * 100
    ttft_gain = (baseline.avg_ttft_ms - best.avg_ttft_ms) / max(baseline.avg_ttft_ms, 1e-9) * 100
    vram_gain = (baseline.peak_vram_mb - best.peak_vram_mb) / max(baseline.peak_vram_mb, 1e-9) * 100

    console.print(
        f"\n[bold green]Best run (ID {best.id}):[/bold green] "
        f"{acc_gain:+.1f}% accuracy  |  "
        f"{ttft_gain:+.1f}% TTFT  |  "
        f"{vram_gain:+.1f}% VRAM vs baseline"
    )


if __name__ == "__main__":
    main()
