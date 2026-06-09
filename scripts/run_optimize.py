#!/usr/bin/env python
"""CLI: full optimization sweep (LR × epochs + quant levels)."""

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
    parser = argparse.ArgumentParser(description="Run full LLM optimization sweep")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-4, 2e-4])
    parser.add_argument("--epochs", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--quant-levels", nargs="+", default=["Q4_K_M", "Q8_0"])
    parser.add_argument("--subjects", nargs="+", default=["high_school_mathematics"])
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-benchmark-samples", type=int, default=50)
    parser.add_argument("--vram-budget-mb", type=float, default=None)
    parser.add_argument("--latency-weight", type=float, default=0.3)
    parser.add_argument("--memory-weight", type=float, default=0.2)
    parser.add_argument("--adapters-dir", default="./adapters")
    parser.add_argument("--models-dir", default="./models")
    parser.add_argument("--llama-cpp-dir", default=None)
    parser.add_argument(
        "--mode",
        choices=["full", "finetune-only", "quant-only"],
        default="full",
        help="full = all phases; finetune-only = skip GGUF; quant-only = skip training",
    )
    parser.add_argument("--adapter", default=None, help="Existing adapter for quant-only mode")
    args = parser.parse_args()

    from src.optimize.pipeline import OptimizationConfig, OptimizationPipeline
    from src.storage.db import init_db

    init_db()

    cfg = OptimizationConfig(
        base_model_id=args.base_model,
        learning_rates=args.learning_rates,
        epochs_list=args.epochs,
        quant_levels=args.quant_levels,
        mmlu_subjects=args.subjects,
        max_samples_per_subject=args.max_benchmark_samples,
        max_train_samples=args.max_train_samples,
        vram_budget_mb=args.vram_budget_mb,
        latency_weight=args.latency_weight,
        memory_weight=args.memory_weight,
        adapters_dir=args.adapters_dir,
        models_dir=args.models_dir,
        llama_cpp_dir=args.llama_cpp_dir,
    )

    pipeline = OptimizationPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
