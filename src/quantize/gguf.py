from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def merge_and_export(
    base_model_id: str,
    adapter_path: str,
    output_hf_dir: str,
) -> str:
    """Merge LoRA adapter into base model and save merged HF weights."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model {base_model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print(f"Loading adapter {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter...")
    model = model.merge_and_unload()

    out = Path(output_hf_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"Merged model saved to {out}")
    return str(out)


def convert_to_gguf(hf_model_dir: str, output_path: str, llama_cpp_dir: str | None = None) -> str:
    """Convert HF model directory to GGUF f16 format via llama.cpp script."""
    script = _find_convert_script(llama_cpp_dir)
    output_path = str(Path(output_path).with_suffix(".gguf"))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, script, hf_model_dir, "--outtype", "f16", "--outfile", output_path]
    print(f"Converting to GGUF: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed:\n{result.stderr}")

    return output_path


def quantize_gguf(
    f16_gguf_path: str,
    quant_level: str,
    output_path: str,
    llama_cpp_dir: str | None = None,
) -> str:
    """Quantize a GGUF f16 file to the given quant level."""
    quantize_bin = _find_quantize_bin(llama_cpp_dir)
    output_path = str(Path(output_path).with_suffix(".gguf"))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [quantize_bin, f16_gguf_path, output_path, quant_level]
    print(f"Quantizing {quant_level}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Quantization failed:\n{result.stderr}")

    return output_path


def run_quant_sweep(
    hf_model_dir: str,
    quant_levels: list[str],
    output_dir: str,
    llama_cpp_dir: str | None = None,
) -> list[str]:
    """Convert HF → GGUF f16, then quantize to each quant level. Returns output paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f16_path = str(out_dir / "model-f16.gguf")
    convert_to_gguf(hf_model_dir, f16_path, llama_cpp_dir)

    output_paths: list[str] = []
    for level in quant_levels:
        q_path = str(out_dir / f"model-{level}.gguf")
        quantize_gguf(f16_path, level, q_path, llama_cpp_dir)
        output_paths.append(q_path)

    return output_paths


# ---------------------------------------------------------------------------


def _find_convert_script(llama_cpp_dir: str | None) -> str:
    candidates = []
    if llama_cpp_dir:
        candidates.append(str(Path(llama_cpp_dir) / "convert_hf_to_gguf.py"))
    candidates += [
        "convert_hf_to_gguf.py",
        "./llama.cpp/convert_hf_to_gguf.py",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError(
        "convert_hf_to_gguf.py not found. Clone llama.cpp and set LLAMA_CPP_DIR."
    )


def _find_quantize_bin(llama_cpp_dir: str | None) -> str:
    candidates = []
    if llama_cpp_dir:
        candidates += [
            str(Path(llama_cpp_dir) / "llama-quantize"),
            str(Path(llama_cpp_dir) / "llama-quantize.exe"),
            str(Path(llama_cpp_dir) / "build" / "bin" / "llama-quantize"),
        ]
    candidates += ["llama-quantize", "llama-quantize.exe", "./llama.cpp/build/bin/llama-quantize"]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "llama-quantize binary not found. Build llama.cpp and add it to PATH or set LLAMA_CPP_DIR."
    )
