from dataclasses import dataclass

from src.optimize.pipeline import OptimizationConfig, composite_score


@dataclass
class FakeRun:
    accuracy: float
    avg_ttft_ms: float
    peak_vram_mb: float
    model_name: str = "test"
    adapter_name: str | None = None
    quant_level: str | None = None


def test_composite_score_accuracy_dominates():
    baseline = FakeRun(accuracy=0.30, avg_ttft_ms=400, peak_vram_mb=1000)
    better_acc = FakeRun(accuracy=0.35, avg_ttft_ms=400, peak_vram_mb=1000)
    score = composite_score(better_acc, baseline)
    assert score > 0


def test_composite_score_ttft_penalty():
    baseline = FakeRun(accuracy=0.30, avg_ttft_ms=400, peak_vram_mb=1000)
    slower = FakeRun(accuracy=0.30, avg_ttft_ms=600, peak_vram_mb=1000)
    score = composite_score(slower, baseline)
    assert score < 0


def test_composite_score_vram_penalty():
    baseline = FakeRun(accuracy=0.30, avg_ttft_ms=400, peak_vram_mb=1000)
    more_vram = FakeRun(accuracy=0.30, avg_ttft_ms=400, peak_vram_mb=2000)
    score = composite_score(more_vram, baseline)
    assert score < 0


def test_composite_score_faster_wins_even_at_flat_accuracy():
    baseline = FakeRun(accuracy=0.30, avg_ttft_ms=400, peak_vram_mb=1000)
    faster = FakeRun(accuracy=0.30, avg_ttft_ms=300, peak_vram_mb=700)
    score = composite_score(faster, baseline)
    assert score > 0


def test_optimization_config_defaults():
    cfg = OptimizationConfig(base_model_id="test-model")
    assert cfg.latency_weight == 0.3
    assert cfg.memory_weight == 0.2
    assert "Q4_K_M" in cfg.quant_levels
    assert len(cfg.learning_rates) >= 1
