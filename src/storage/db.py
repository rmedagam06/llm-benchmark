from __future__ import annotations

import json
import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.storage.schema import Base, BenchmarkRun, TrainingRun

_engine = None


def _get_db_path() -> str:
    return os.environ.get("DB_PATH", "./data/results/bench.db")


def init_db(db_path: str | None = None) -> None:
    global _engine
    path = db_path or _get_db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(f"sqlite:///{path}", echo=False)
    Base.metadata.create_all(_engine)


def _ensure_engine() -> None:
    if _engine is None:
        init_db()


# ---------------------------------------------------------------------------


def save_benchmark(result) -> int:
    """Save a BenchmarkResult (from runner.py) and return the row id."""
    _ensure_engine()
    with Session(_engine) as session:
        row = BenchmarkRun(
            model_name=result.model_name,
            adapter_name=result.adapter_name,
            quant_level=result.quant_level,
            accuracy=result.accuracy,
            per_subject_json=json.dumps(result.per_subject_accuracy),
            avg_ttft_ms=result.avg_ttft_ms,
            tokens_per_second=result.tokens_per_second,
            peak_vram_mb=result.peak_vram_mb,
            num_questions=result.num_questions,
            subjects_json=json.dumps(result.subjects),
            timestamp=result.timestamp,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id


def save_training(metrics) -> int:
    """Save a TrainingMetrics object and return the row id."""
    _ensure_engine()
    with Session(_engine) as session:
        row = TrainingRun(
            model_name=metrics.model_name,
            adapter_path=metrics.adapter_name,
            learning_rate=metrics.learning_rate,
            epochs=metrics.epochs,
            training_time_seconds=metrics.training_time_seconds,
            final_loss=metrics.final_loss,
            peak_vram_mb=metrics.peak_vram_mb,
            seed=metrics.seed,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id


def get_all_benchmarks(model_name: str | None = None) -> list[BenchmarkRun]:
    _ensure_engine()
    with Session(_engine) as session:
        q = session.query(BenchmarkRun)
        if model_name:
            q = q.filter(BenchmarkRun.model_name == model_name)
        return q.order_by(BenchmarkRun.id).all()


def get_best_benchmark(metric: str = "accuracy") -> BenchmarkRun | None:
    _ensure_engine()
    with Session(_engine) as session:
        col = getattr(BenchmarkRun, metric, BenchmarkRun.accuracy)
        return session.query(BenchmarkRun).order_by(col.desc()).first()


def compare(baseline_id: int, candidate_id: int) -> dict:
    _ensure_engine()
    with Session(_engine) as session:
        baseline = session.get(BenchmarkRun, baseline_id)
        candidate = session.get(BenchmarkRun, candidate_id)
        if not baseline or not candidate:
            return {}
        return {
            "accuracy_delta": candidate.accuracy - baseline.accuracy,
            "accuracy_delta_pct": (candidate.accuracy - baseline.accuracy)
            / max(baseline.accuracy, 1e-9)
            * 100,
            "ttft_delta_ms": candidate.avg_ttft_ms - baseline.avg_ttft_ms,
            "ttft_delta_pct": (candidate.avg_ttft_ms - baseline.avg_ttft_ms)
            / max(baseline.avg_ttft_ms, 1e-9)
            * 100,
            "vram_delta_mb": candidate.peak_vram_mb - baseline.peak_vram_mb,
            "vram_delta_pct": (candidate.peak_vram_mb - baseline.peak_vram_mb)
            / max(baseline.peak_vram_mb, 1e-9)
            * 100,
        }
