from unittest.mock import MagicMock, patch

import pytest

from src.quantize.gguf import convert_to_gguf, quantize_gguf


def test_convert_to_gguf_calls_subprocess(tmp_path):
    fake_script = tmp_path / "convert_hf_to_gguf.py"
    fake_script.write_text("# fake")
    f16_out = str(tmp_path / "model-f16.gguf")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = convert_to_gguf(str(tmp_path / "hf_model"), f16_out, llama_cpp_dir=str(tmp_path))

    assert mock_run.called
    assert result.endswith(".gguf")


def test_quantize_gguf_calls_subprocess(tmp_path):
    fake_bin = tmp_path / "llama-quantize"
    fake_bin.write_text("#!/bin/bash\nexit 0")
    f16 = tmp_path / "model-f16.gguf"
    f16.write_text("fake")
    q_out = str(tmp_path / "model-Q4_K_M.gguf")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = quantize_gguf(str(f16), "Q4_K_M", q_out, llama_cpp_dir=str(tmp_path))

    assert mock_run.called
    assert "Q4_K_M" in result


def test_quantize_gguf_raises_on_failure(tmp_path):
    fake_bin = tmp_path / "llama-quantize"
    fake_bin.write_text("#!/bin/bash\nexit 1")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="quantize error")
        with pytest.raises(RuntimeError, match="Quantization failed"):
            quantize_gguf("model.gguf", "Q4_K_M", str(tmp_path / "out.gguf"), str(tmp_path))
