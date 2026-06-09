import pytest

from src.finetune.lora import LoRAConfig, _format_sample


def test_lora_config_defaults():
    cfg = LoRAConfig(base_model_id="gpt2")
    assert cfg.r == 16
    assert cfg.seed == 42
    assert cfg.max_samples == 3000
    assert "q_proj" in cfg.target_modules


def test_format_sample_instruction_output():
    sample = {"instruction": "Explain LoRA", "output": "LoRA is..."}
    text = _format_sample(sample)
    assert "<|user|>" in text
    assert "<|assistant|>" in text
    assert "Explain LoRA" in text
    assert "LoRA is..." in text


def test_format_sample_conversations():
    sample = {
        "conversations": [
            {"role": "user", "value": "Hello"},
            {"role": "assistant", "value": "Hi there"},
        ]
    }
    text = _format_sample(sample)
    assert "Hello" in text
    assert "Hi there" in text


@pytest.mark.slow
def test_train_lora_smoke(tmp_path):
    """Smoke test: 2 steps on gpt2. Requires torch."""
    pytest.skip("Skipping GPU-dependent smoke test in standard run")
