from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from src.benchmark.mmlu import extract_answer, load_mmlu_subset

console = Console()


@dataclass
class BenchmarkConfig:
    model_name: str
    subjects: list[str]
    num_shots: int = 0
    max_samples: int = 50
    max_new_tokens: int = 8
    temperature: float = 0.0
    adapter_name: str | None = None
    quant_level: str | None = None


@dataclass
class BenchmarkResult:
    model_name: str
    adapter_name: str | None
    quant_level: str | None
    subjects: list[str]
    num_shots: int
    num_questions: int
    num_correct: int
    accuracy: float
    per_subject_accuracy: dict[str, float]
    avg_ttft_ms: float
    tokens_per_second: float
    peak_vram_mb: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "adapter_name": self.adapter_name,
            "quant_level": self.quant_level,
            "subjects": self.subjects,
            "num_shots": self.num_shots,
            "num_questions": self.num_questions,
            "num_correct": self.num_correct,
            "accuracy": self.accuracy,
            "per_subject_accuracy": self.per_subject_accuracy,
            "avg_ttft_ms": self.avg_ttft_ms,
            "tokens_per_second": self.tokens_per_second,
            "peak_vram_mb": self.peak_vram_mb,
            "timestamp": self.timestamp,
        }


def run_benchmark(engine, config: BenchmarkConfig) -> BenchmarkResult:
    """Run MMLU benchmark and return results. Saves JSON to data/results/."""
    items = load_mmlu_subset(
        subjects=config.subjects,
        num_shots=config.num_shots,
        max_samples=config.max_samples,
    )

    ttft_list: list[float] = []
    tps_list: list[float] = []
    peak_vram_list: list[float] = []
    per_subject: dict[str, list[int]] = {s: [] for s in config.subjects}

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Benchmarking {config.model_name}", total=len(items))

        for item in items:
            text, metrics = engine.generate_with_metrics(
                item["prompt"],
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
            predicted = extract_answer(text)
            correct = int(predicted == item["answer_label"]) if predicted else 0
            per_subject[item["subject"]].append(correct)

            ttft_list.append(metrics.ttft_ms)
            tps_list.append(metrics.tokens_per_second)
            peak_vram_list.append(metrics.peak_vram_mb)
            progress.advance(task)

    per_subject_accuracy = {s: sum(v) / len(v) if v else 0.0 for s, v in per_subject.items()}
    all_correct = sum(sum(v) for v in per_subject.values())
    accuracy = all_correct / len(items) if items else 0.0

    result = BenchmarkResult(
        model_name=config.model_name,
        adapter_name=config.adapter_name,
        quant_level=config.quant_level,
        subjects=config.subjects,
        num_shots=config.num_shots,
        num_questions=len(items),
        num_correct=all_correct,
        accuracy=accuracy,
        per_subject_accuracy=per_subject_accuracy,
        avg_ttft_ms=sum(ttft_list) / len(ttft_list) if ttft_list else 0.0,
        tokens_per_second=sum(tps_list) / len(tps_list) if tps_list else 0.0,
        peak_vram_mb=max(peak_vram_list, default=0.0),
    )

    _save_json(result, config)
    _save_to_db(result)
    return result


def _save_json(result: BenchmarkResult, config: BenchmarkConfig) -> None:
    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = result.model_name.replace("/", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"{safe_name}_{ts}.json"
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    console.print(f"[dim]Results saved to {path}[/dim]")


def _save_to_db(result: BenchmarkResult) -> None:
    try:
        from src.storage.db import save_benchmark

        save_benchmark(result)
    except Exception:
        pass  # DB not yet initialised — skip silently
