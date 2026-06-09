from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    adapter_name: Mapped[str | None] = mapped_column(String, nullable=True)
    quant_level: Mapped[str | None] = mapped_column(String, nullable=True)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    per_subject_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    avg_ttft_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    tokens_per_second: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    peak_vram_mb: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    num_questions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    subjects_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    timestamp: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default=lambda: datetime.now(timezone.utc).isoformat(),
    )


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    adapter_path: Mapped[str] = mapped_column(String, nullable=False)
    learning_rate: Mapped[float] = mapped_column(Float, nullable=False)
    epochs: Mapped[int] = mapped_column(Integer, nullable=False)
    training_time_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    final_loss: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    peak_vram_mb: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    seed: Mapped[int] = mapped_column(Integer, nullable=False, default=42)
    created_at: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default=lambda: datetime.now(timezone.utc).isoformat(),
    )
