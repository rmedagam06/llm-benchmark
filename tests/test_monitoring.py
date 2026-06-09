import time

from src.monitoring.metrics import InferenceMetrics, TrainingMetrics, TTFTTimer, VRAMMonitor


def test_vram_monitor_produces_samples():
    m = VRAMMonitor()
    m.start()
    time.sleep(0.35)
    m.stop()
    assert m.peak_vram_mb > 0
    assert m.avg_vram_mb > 0


def test_vram_monitor_context_manager():
    with VRAMMonitor() as m:
        time.sleep(0.25)
    assert m.peak_vram_mb > 0


def test_ttft_timer():
    t = TTFTTimer()
    t.mark_request()
    time.sleep(0.05)
    t.mark_first_token()
    assert t.ttft_ms >= 45.0  # at least 45 ms


def test_ttft_timer_unset_returns_zero():
    t = TTFTTimer()
    assert t.ttft_ms == 0.0


def test_inference_metrics_serialise():
    m = InferenceMetrics(
        model_name="gpt2",
        adapter_name=None,
        ttft_ms=12.5,
        tokens_per_second=45.2,
        peak_vram_mb=512.0,
        total_tokens=32,
    )
    d = m.to_dict()
    assert d["model_name"] == "gpt2"
    assert d["adapter_name"] is None
    assert d["ttft_ms"] == 12.5


def test_training_metrics_serialise():
    m = TrainingMetrics(
        model_name="gpt2",
        adapter_name="gpt2-lora",
        learning_rate=2e-4,
        epochs=1,
        training_time_seconds=120.0,
        final_loss=1.23,
        peak_vram_mb=4096.0,
        seed=42,
    )
    d = m.to_dict()
    assert d["learning_rate"] == 2e-4
    assert d["seed"] == 42
