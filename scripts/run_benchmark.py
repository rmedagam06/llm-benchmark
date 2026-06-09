#!/usr/bin/env python
"""CLI: benchmark a model on MMLU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # optional

try:
    load_dotenv()
except ImportError:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an HF or GGUF model on MMLU")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or GGUF file path")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path")
    parser.add_argument(
        "--gguf", default=None, help="GGUF model file path (use instead of --model)"
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["high_school_mathematics"],
        help="MMLU subjects",
    )
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--num-shots", type=int, default=0)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    from src.storage.db import init_db

    init_db()

    if args.gguf:
        from src.inference.gguf_engine import GGUFEngine

        engine = GGUFEngine(args.gguf)
        model_name = Path(args.gguf).stem
        quant_level = None
        for q in ["Q4_K_M", "Q5_K_M", "Q8_0", "Q4_0", "Q8"]:
            if q in model_name.upper():
                quant_level = q
                break
    else:
        from src.inference.hf_engine import HFEngine

        engine = HFEngine(args.model, load_in_4bit=not args.no_4bit)
        model_name = args.model
        quant_level = None

        if args.adapter:
            engine.load_adapter(args.adapter)

    from rich.console import Console

    from src.benchmark.runner import BenchmarkConfig, run_benchmark

    console = Console()
    config = BenchmarkConfig(
        model_name=model_name,
        adapter_name=args.adapter,
        quant_level=quant_level,
        subjects=args.subjects,
        num_shots=args.num_shots,
        max_samples=args.max_samples,
    )

    result = run_benchmark(engine, config)

    console.print(f"\n[bold green]Accuracy:[/bold green] {result.accuracy:.1%}")
    console.print(f"[bold]Avg TTFT:[/bold] {result.avg_ttft_ms:.1f} ms")
    console.print(f"[bold]Tokens/sec:[/bold] {result.tokens_per_second:.1f}")
    console.print(f"[bold]Peak VRAM:[/bold] {result.peak_vram_mb:.0f} MB")
    for subj, acc in result.per_subject_accuracy.items():
        console.print(f"  {subj}: {acc:.1%}")


if __name__ == "__main__":
    main()
