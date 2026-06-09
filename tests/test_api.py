from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.monitoring.metrics import InferenceMetrics


@pytest.fixture(scope="module")
def client():
    with patch("src.api.main.lifespan"):
        from src.api.main import app

        with TestClient(app) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "vram_mb" in data
    assert "uptime_s" in data


def test_models(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "loaded_hf" in data
    assert "gguf_files" in data


def test_results(client):
    resp = client.get("/results")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_infer_model_not_found(client):
    resp = client.post(
        "/infer",
        json={"model": "nonexistent", "prompt": "hello", "max_tokens": 10},
    )
    assert resp.status_code == 404


def test_infer_non_streaming(client):
    fake_metrics = InferenceMetrics(
        model_name="gpt2",
        adapter_name=None,
        ttft_ms=10.0,
        tokens_per_second=50.0,
        peak_vram_mb=256.0,
        total_tokens=5,
    )
    fake_engine = MagicMock()
    fake_engine.generate_with_metrics.return_value = ("some generated text", fake_metrics)

    from src.inference.hf_engine import ModelRegistry

    ModelRegistry._engines["gpt2"] = fake_engine

    resp = client.post(
        "/infer",
        json={"model": "gpt2", "prompt": "Hello", "max_tokens": 10, "stream": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
    assert "metrics" in data
    assert data["metrics"]["ttft_ms"] == 10.0

    ModelRegistry._engines.pop("gpt2", None)


def test_compare_not_found(client):
    resp = client.get("/results/compare?baseline_id=9999&candidate_id=9998")
    assert resp.status_code == 404
