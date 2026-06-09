from unittest.mock import MagicMock, patch

import pytest
import torch

from src.inference.hf_engine import HFEngine, ModelRegistry
from src.monitoring.metrics import InferenceMetrics


def _make_mock_tokenizer():
    tok = MagicMock()
    tok.eos_token = "<eos>"
    tok.pad_token = None
    tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    tok.decode.return_value = " hello world"
    tok.encode.return_value = [1, 2, 3, 4, 5]
    return tok


def _make_mock_model():
    model = MagicMock()
    param = MagicMock()
    param.device = torch.device("cpu")
    model.parameters.return_value = iter([param])
    model.to.return_value = model
    model.eval.return_value = model
    # non-streaming generate: return a batch of token ids (input 3 tokens + 4 new)
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    return model


@pytest.fixture
def engine():
    mock_tok = _make_mock_tokenizer()
    mock_model = _make_mock_model()
    with (
        patch("src.inference.hf_engine.AutoTokenizer") as tok_cls,
        patch("src.inference.hf_engine.AutoModelForCausalLM") as model_cls,
    ):
        tok_cls.from_pretrained.return_value = mock_tok
        model_cls.from_pretrained.return_value = mock_model
        e = HFEngine("gpt2", load_in_4bit=False)
    yield e


def test_engine_loads(engine):
    assert engine.model is not None
    assert engine.tokenizer is not None


def test_generate_no_stream(engine):
    tokens = list(engine.generate("Hello world", max_new_tokens=5, stream=False))
    assert len(tokens) == 1
    assert isinstance(tokens[0], str)


def test_generate_yields_tokens(engine):
    # Patch TextIteratorStreamer to yield predefined tokens without needing a real model.
    mock_streamer = MagicMock()
    mock_streamer.__iter__ = MagicMock(return_value=iter(["hello", " world"]))
    with patch("src.inference.hf_engine.TextIteratorStreamer", return_value=mock_streamer):
        tokens = list(engine.generate("The future of AI is", max_new_tokens=10, stream=True))
    assert len(tokens) >= 1
    assert any(t.strip() for t in tokens)


def test_generate_with_metrics(engine):
    mock_streamer = MagicMock()
    mock_streamer.__iter__ = MagicMock(return_value=iter(["The", " meaning", " is", " 42"]))
    with patch("src.inference.hf_engine.TextIteratorStreamer", return_value=mock_streamer):
        text, metrics = engine.generate_with_metrics("The meaning of life is", max_new_tokens=15)
    assert isinstance(metrics, InferenceMetrics)
    assert metrics.model_name == "gpt2"
    assert metrics.ttft_ms >= 0
    assert metrics.tokens_per_second >= 0
    assert metrics.peak_vram_mb >= 0
    assert len(text) > 0


def test_model_registry():
    mock_tok = _make_mock_tokenizer()
    mock_model = _make_mock_model()
    ModelRegistry._engines.clear()
    with (
        patch("src.inference.hf_engine.AutoTokenizer") as tok_cls,
        patch("src.inference.hf_engine.AutoModelForCausalLM") as model_cls,
    ):
        tok_cls.from_pretrained.return_value = mock_tok
        model_cls.from_pretrained.return_value = mock_model
        engine = ModelRegistry.load("test-gpt2", "gpt2", load_in_4bit=False)
    assert ModelRegistry.get("test-gpt2") is engine
    ModelRegistry.unload_all()
    assert ModelRegistry.get("test-gpt2") is None
