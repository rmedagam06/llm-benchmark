#!/usr/bin/env python
"""CLI: fine-tune a model with QLoRA."""

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
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning on instruction data")
    parser.add_argument("--base-model", required=True, help="Base HF model ID")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", default="./adapters")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from rich.console import Console

    from src.finetune.lora import LoRAConfig, train_lora
    from src.storage.db import init_db

    console = Console()
    init_db()

    cfg = LoRAConfig(
        base_model_id=args.base_model,
        max_samples=args.max_samples,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    console.print(
        f"[bold]Training:[/bold] lr={args.learning_rate}, epochs={args.epochs}, samples={args.max_samples}"
    )
    metrics, adapter_path = train_lora(cfg)

    console.print("\n[bold green]Training complete![/bold green]")
    console.print(f"Adapter saved to: {adapter_path}")
    console.print(f"Final loss: {metrics.final_loss:.4f}")
    console.print(f"Training time: {metrics.training_time_seconds:.0f}s")
    console.print(f"Peak VRAM: {metrics.peak_vram_mb:.0f} MB")


if __name__ == "__main__":
    main()
