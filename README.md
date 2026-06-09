# LLM Benchmark

A platform for evaluating and optimizing open-source language models. It runs accuracy benchmarks, fine-tunes models on instruction data, tests different compression levels, and picks the best configuration based on a weighted score across accuracy, speed, and memory usage.

---

## What it does

Most LLM optimization workflows are manual — you pick a model, maybe quantize it, and eyeball the results. This automates the full cycle:

1. **Baseline benchmark** — runs the model on MMLU (a standard multiple-choice academic benchmark) to get a starting accuracy, response speed, and VRAM number
2. **Fine-tuning sweep** — trains LoRA adapters on OpenHermes-2.5 instruction data across multiple learning rates and epoch counts
3. **Quantization sweep** — exports the best adapter to GGUF format and tests Q4, Q5, and Q8 compression levels
4. **Winner selection** — scores every configuration using a composite formula that weighs accuracy improvement against latency and memory cost, then prints a comparison table

The end result is a table showing exactly how each configuration performed and which one won, with the reasoning explained.

---

## Stack

- **Inference** — HuggingFace Transformers with 4-bit quantization via bitsandbytes
- **Fine-tuning** — QLoRA via PEFT and TRL's SFTTrainer
- **Quantization** — llama.cpp GGUF export (Q4_K_M, Q5_K_M, Q8_0)
- **Benchmarking** — MMLU dataset (0-shot and 5-shot)
- **Storage** — SQLite via SQLAlchemy
- **API** — FastAPI with server-sent event streaming
- **Monitoring** — real-time VRAM and TTFT tracking per inference call

---

## Project structure

```
src/
├── inference/       # HuggingFace and GGUF inference engines
├── benchmark/       # MMLU loader and benchmark runner
├── finetune/        # QLoRA training with SFTTrainer
├── quantize/        # GGUF export and quantization sweep
├── optimize/        # Full pipeline + composite scoring
├── storage/         # SQLite schema and query helpers
└── api/             # FastAPI endpoints

scripts/
├── run_benchmark.py  # Benchmark a single model
├── run_finetune.py   # Fine-tune with QLoRA
├── run_optimize.py   # Run the full sweep
└── report.py         # Print results table from DB
```

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows
# source .venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
cp .env.example .env
```

---

## Running it

**Benchmark a model:**
```bash
python scripts/run_benchmark.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --subjects high_school_mathematics \
  --max-samples 100
```

**Run the full optimization sweep:**
```bash
python scripts/run_optimize.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --subjects high_school_mathematics \
  --learning-rates 1e-4 2e-4 \
  --epochs 1 2 \
  --max-train-samples 3000
```

**Print the results table:**
```bash
python scripts/report.py
```

**Start the API server:**
```bash
python scripts/serve.py
```

---

## Composite scoring

Each configuration is scored against the baseline using:

```
score = accuracy_gain
      - (0.3 × ttft_penalty)
      - (0.2 × vram_penalty)
```

A configuration that cuts VRAM by 40% and speeds up response time by 25% can outscore one with a slightly higher accuracy — which is usually the right call for deployment.

---

## Requirements

- Python 3.10+
- CUDA GPU with 8GB+ VRAM for 3B models in 4-bit
- For GGUF quantization: llama.cpp built locally, path passed via `--llama-cpp-dir`
