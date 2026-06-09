from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class OptimizationConfig:
    base_model_id: str
    learning_rates: list[float] = field(default_factory=lambda: [1e-4, 2e-4, 5e-4])
    epochs_list: list[int] = field(default_factory=lambda: [1, 2])
    quant_levels: list[str] = field(default_factory=lambda: ["Q4_K_M", "Q5_K_M", "Q8_0"])
    mmlu_subjects: list[str] = field(default_factory=lambda: ["high_school_mathematics"])
    max_samples_per_subject: int = 50
    max_train_samples: int = 2000
    vram_budget_mb: float | None = None
    latency_weight: float = 0.3
    memory_weight: float = 0.2
    adapters_dir: str = "./adapters"
    models_dir: str = "./models"
    llama_cpp_dir: str | None = None


@dataclass
class OptimizationResult:
    baseline: Any  # BenchmarkRun
    winner: Any  # BenchmarkRun
    winner_training: Any  # TrainingRun or None
    accuracy_delta_pct: float
    ttft_delta_pct: float
    vram_delta_pct: float
    composite_score: float
    selected_lr: float | None
    selected_epochs: int | None
    selected_quant: str | None
    selection_reason: str
    all_runs: list[Any]


def composite_score(
    run,
    baseline,
    latency_weight: float = 0.3,
    memory_weight: float = 0.2,
) -> float:
    accuracy_gain = run.accuracy - baseline.accuracy
    ttft_penalty = (run.avg_ttft_ms - baseline.avg_ttft_ms) / max(baseline.avg_ttft_ms, 1e-9)
    vram_penalty = (run.peak_vram_mb - baseline.peak_vram_mb) / max(baseline.peak_vram_mb, 1e-9)
    return accuracy_gain - (latency_weight * ttft_penalty) - (memory_weight * vram_penalty)


class OptimizationPipeline:
    def __init__(self, config: OptimizationConfig) -> None:
        self.cfg = config
        self._training_map: dict[str, Any] = {}  # adapter_path → TrainingMetrics

    # ------------------------------------------------------------------

    def run(self) -> OptimizationResult:
        from src.benchmark.runner import BenchmarkConfig, run_benchmark
        from src.finetune.lora import LoRAConfig, train_lora
        from src.inference.hf_engine import HFEngine

        cfg = self.cfg

        # Step A — baseline
        console.rule("[bold]Step A: Baseline benchmark")
        base_engine = HFEngine(cfg.base_model_id, load_in_4bit=True)
        run_benchmark(
            base_engine,
            BenchmarkConfig(
                model_name=cfg.base_model_id,
                subjects=cfg.mmlu_subjects,
                max_samples=cfg.max_samples_per_subject,
            ),
        )
        base_engine.unload()

        from src.storage.db import get_all_benchmarks

        baseline_row = get_all_benchmarks()[-1]  # most recently saved

        all_runs = [baseline_row]
        lora_rows: list[Any] = []

        # Step B — LR × epochs sweep
        console.rule("[bold]Step B: LoRA fine-tuning sweep")
        for lr in cfg.learning_rates:
            for epochs in cfg.epochs_list:
                console.print(f"Training: lr={lr}, epochs={epochs}")
                lora_cfg = LoRAConfig(
                    base_model_id=cfg.base_model_id,
                    max_samples=cfg.max_train_samples,
                    learning_rate=lr,
                    num_epochs=epochs,
                    output_dir=cfg.adapters_dir,
                )
                metrics, adapter_path = train_lora(lora_cfg)
                self._training_map[adapter_path] = metrics

                engine = HFEngine(cfg.base_model_id, load_in_4bit=True)
                engine.load_adapter(adapter_path)
                run_benchmark(
                    engine,
                    BenchmarkConfig(
                        model_name=cfg.base_model_id,
                        adapter_name=adapter_path,
                        subjects=cfg.mmlu_subjects,
                        max_samples=cfg.max_samples_per_subject,
                    ),
                )
                engine.unload()

                rows = get_all_benchmarks()
                row = rows[-1]
                all_runs.append(row)
                lora_rows.append((row, adapter_path, lr, epochs))

        # Step C — select best adapter
        console.rule("[bold]Step C: Selecting best adapter")
        best_row, best_adapter, best_lr, best_epochs = max(
            lora_rows,
            key=lambda x: composite_score(
                x[0], baseline_row, cfg.latency_weight, cfg.memory_weight
            ),
        )

        # Step D — export best adapter to GGUF
        console.rule("[bold]Step D: GGUF export")
        from src.quantize.gguf import merge_and_export, run_quant_sweep

        merged_dir = str(Path(cfg.models_dir) / "merged")
        merge_and_export(cfg.base_model_id, best_adapter, merged_dir)

        quant_dir = str(Path(cfg.models_dir) / "quantized")
        gguf_paths = run_quant_sweep(merged_dir, cfg.quant_levels, quant_dir, cfg.llama_cpp_dir)

        # Step E — benchmark each GGUF
        console.rule("[bold]Step E: GGUF benchmarks")
        from src.inference.gguf_engine import GGUFEngine

        gguf_rows = []
        for gguf_path, level in zip(gguf_paths, cfg.quant_levels):
            if cfg.vram_budget_mb:
                pass  # checked post-run below

            g_engine = GGUFEngine(gguf_path)
            run_benchmark(
                g_engine,
                BenchmarkConfig(
                    model_name=Path(gguf_path).stem,
                    quant_level=level,
                    subjects=cfg.mmlu_subjects,
                    max_samples=cfg.max_samples_per_subject,
                ),
            )
            g_engine.unload()

            rows = get_all_benchmarks()
            row = rows[-1]

            if cfg.vram_budget_mb and row.peak_vram_mb > cfg.vram_budget_mb:
                console.print(
                    f"[yellow]Skipping {level} — exceeds VRAM budget ({row.peak_vram_mb:.0f} MB)[/yellow]"
                )
                continue

            all_runs.append(row)
            gguf_rows.append((row, level))

        # Step F — select overall winner
        console.rule("[bold]Step F: Selecting winner")
        all_candidates = [(r, lvl) for (r, _, _, _) in lora_rows for lvl in [None]] + gguf_rows

        winner_row, winner_quant = max(
            all_candidates,
            key=lambda x: composite_score(
                x[0], baseline_row, cfg.latency_weight, cfg.memory_weight
            ),
        )

        score = composite_score(winner_row, baseline_row, cfg.latency_weight, cfg.memory_weight)
        acc_delta = (winner_row.accuracy - baseline_row.accuracy) * 100
        ttft_delta = (
            (winner_row.avg_ttft_ms - baseline_row.avg_ttft_ms)
            / max(baseline_row.avg_ttft_ms, 1e-9)
            * 100
        )
        vram_delta = (
            (winner_row.peak_vram_mb - baseline_row.peak_vram_mb)
            / max(baseline_row.peak_vram_mb, 1e-9)
            * 100
        )

        reason = (
            f"best composite score {score:+.4f}: "
            f"{acc_delta:+.1f}% accuracy, "
            f"{ttft_delta:+.1f}% TTFT, "
            f"{vram_delta:+.1f}% VRAM"
        )

        winner_training = self._training_map.get(best_adapter)

        result = OptimizationResult(
            baseline=baseline_row,
            winner=winner_row,
            winner_training=winner_training,
            accuracy_delta_pct=acc_delta,
            ttft_delta_pct=ttft_delta,
            vram_delta_pct=vram_delta,
            composite_score=score,
            selected_lr=best_lr,
            selected_epochs=best_epochs,
            selected_quant=winner_quant,
            selection_reason=reason,
            all_runs=all_runs,
        )

        # Step G — print table
        _print_table(result, cfg.base_model_id)
        return result


# ---------------------------------------------------------------------------


def _run_label(row, base_model_id: str) -> str:
    if row.adapter_name is None and row.quant_level is None:
        return "Baseline (no adapter)"
    if row.quant_level:
        return f"GGUF {row.quant_level}"
    if row.adapter_name:
        return "LoRA adapter"
    return row.model_name


def _print_table(result: OptimizationResult, base_model_id: str) -> None:
    table = Table(title=f"LLM OPTIMIZATION REPORT — {base_model_id}", show_lines=True)
    table.add_column("Config", style="cyan", min_width=30)
    table.add_column("Accuracy", justify="right")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("VRAM (MB)", justify="right")

    for row in result.all_runs:
        label = _run_label(row, base_model_id)
        is_winner = row is result.winner
        acc = f"{row.accuracy:.1%}" + (" ▲" if row.accuracy > result.baseline.accuracy else "")
        ttft = f"{row.avg_ttft_ms:.0f}"
        vram = f"{row.peak_vram_mb:.0f}"
        if is_winner:
            label += " ★"
        table.add_row(label, acc, ttft, vram)

    console.print(table)
    console.print(f"[bold green]Winner:[/bold green] {result.selection_reason}")
    console.print(
        f"[bold]Summary:[/bold] "
        f"{result.accuracy_delta_pct:+.1f}% accuracy  |  "
        f"{result.ttft_delta_pct:+.1f}% TTFT  |  "
        f"{result.vram_delta_pct:+.1f}% VRAM"
    )
