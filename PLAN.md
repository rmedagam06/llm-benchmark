# Local LLM Benchmarking & Inference Platform — Plan

## Context

Building an automated LLM optimization platform that benchmarks open-source models, executes LoRA fine-tuning on instruction data, sweeps quantization levels, and selects optimal deployment configurations based on accuracy, latency, and VRAM. The strongest resume deliverable is a concrete comparison table showing measurable improvement over baseline. FastAPI comes last — proven pipeline results come first.

**Key design decisions:**
- HuggingFace-first throughout; GGUF is a final export/comparison step only
- Fine-tune on instruction data (OpenHermes-2.5), evaluate on MMLU — clean benchmark story
- Hyperparameter sweep: learning rate × epochs (for fine-tuning) + quant level (for GGUF comparison)
- TTFT is comparative across configs, not a hard threshold

---

## Project Root

```
C:\Users\ronik\Documents\llm-bench\
```

---

## Tech Stack

| Layer | Library |
|---|---|
| Inference (primary) | `transformers` + `bitsandbytes` (4-bit) + `TextIteratorStreamer` |
| Fine-tuning | `peft` (LoRA/QLoRA) + `trl` (SFTTrainer) |
| Instruction dataset | `datasets` — `teknium/OpenHermes-2.5` |
| MMLU benchmark | `datasets` — `cais/mmlu` |
| VRAM monitoring | `pynvml` + `psutil` |
| GGUF export (Phase 7) | `llama-cpp-python` + `llama.cpp` CLI subprocess |
| API backend | `fastapi` + `uvicorn` + `sse-starlette` |
| Results storage | `SQLite` via `sqlalchemy` |
| CLI reporting | `rich` |
| Testing | `pytest` + `httpx` |
| Config | `pydantic-settings` + `.env` |

---

## Project Structure

```
llm-bench/
├── src/
│   ├── __init__.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── hf_engine.py
│   │   └── gguf_engine.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── mmlu.py
│   │   └── runner.py
│   ├── finetune/
│   │   ├── __init__.py
│   │   └── lora.py
│   ├── quantize/
│   │   ├── __init__.py
│   │   └── gguf.py
│   ├── optimize/
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── routes.py
│   └── storage/
│       ├── __init__.py
│       ├── db.py
│       └── schema.py
├── data/results/
├── models/
├── adapters/
├── tests/
│   ├── test_monitoring.py
│   ├── test_benchmark.py
│   ├── test_inference.py
│   ├── test_finetune.py
│   └── test_api.py
├── scripts/
│   ├── run_benchmark.py
│   ├── run_finetune.py
│   ├── run_optimize.py
│   └── serve.py
├── PLAN.md
├── .env.example
├── requirements.txt
└── pyproject.toml
```

---

## Phase 1 — Project Scaffold & Environment

Create directory structure, requirements.txt, pyproject.toml, .env.example, Python venv, install deps.

**Checkpoint:** `python -c "import torch, transformers, peft, fastapi, pynvml; print('OK')"` + `pytest --collect-only`

---

## Phase 2 — Monitoring Module (`src/monitoring/metrics.py`)

- `VRAMMonitor`: background thread sampling every 100ms, falls back to psutil RSS on CPU-only
- `TTFTTimer`: mark_request / mark_first_token / ttft_ms
- `InferenceMetrics` + `TrainingMetrics` dataclasses

**Checkpoint:** unit tests pass, real numbers print.

---

## Phase 3 — HF Inference Engine (`src/inference/hf_engine.py`)

- `HFEngine`: AutoModelForCausalLM + BitsAndBytesConfig(load_in_4bit=True)
- `load_adapter(adapter_path)` / `unload_adapter()` via PeftModel
- `generate()` with TextIteratorStreamer + VRAMMonitor + TTFTTimer
- `generate_with_metrics()` → tuple[str, InferenceMetrics]
- `ModelRegistry` singleton

**Checkpoint:** gpt2 streams tokens, tests pass.

---

## Phase 4 — MMLU Benchmark Runner (`src/benchmark/`)

- `mmlu.py`: load_mmlu_subset, 0-shot and 5-shot formatting
- `runner.py`: BenchmarkConfig, run_benchmark with rich progress, BenchmarkResult, JSON export to data/results/

**Checkpoint:** real accuracy % prints for 20 questions.

---

## Phase 5 — Results Storage (`src/storage/`)

- `schema.py`: BenchmarkRun + TrainingRun SQLAlchemy ORM
- `db.py`: init_db, save_benchmark, save_training, get_all_benchmarks, get_best_benchmark, compare

**Checkpoint:** stored result prints from DB.

---

## Phase 6 — LoRA Fine-Tuning (`src/finetune/lora.py`)

- `LoRAConfig` dataclass (base_model_id, dataset_name, max_samples=3000, r=16, lora_alpha=32, target_modules, lr, epochs, batch_size, grad_accum, output_dir, max_seq_length, seed)
- `train_lora()`: QLoRA via SFTTrainer, OpenHermes slice `train[:{max_samples}]`
- ChatML format: `<|user|>\n{instruction}\n<|assistant|>\n{output}`
- Saves adapter to `adapters/{name}-lr{lr}-ep{epochs}-{timestamp}/`

**Checkpoint:** adapter files exist, second MMLU benchmark stored in DB.

---

## Phase 7 — GGUF Quantization

- `src/quantize/gguf.py`: merge_and_export, convert_to_gguf, quantize_gguf, run_quant_sweep
- `src/inference/gguf_engine.py`: same interface as HFEngine (transparent to benchmark runner)
- Quant levels: Q4_K_M, Q5_K_M, Q8_0

**Checkpoint:** 3 GGUF files exist, TTFT numbers print.

---

## Phase 8 — Optimization Pipeline (`src/optimize/pipeline.py`) — CORE DELIVERABLE

- `OptimizationConfig`: lr list, epochs list, quant levels, VRAM budget, latency_weight=0.3, memory_weight=0.2
- Composite score: `accuracy_gain - (latency_weight * ttft_penalty) - (memory_weight * vram_penalty)`
- `OptimizationPipeline.run()`: baseline → LR×epochs sweep → select best adapter → quant sweep → select winner
- `OptimizationResult` with selection_reason string
- Final rich comparison table

**Checkpoint:** table prints with real numbers. This IS the resume deliverable.

---

## Phase 9 — FastAPI Backend (`src/api/`)

- `POST /infer` (SSE streaming), `GET /models`, `GET /results`, `GET /results/compare`, `GET /health`

**Checkpoint:** curl health + infer + results all return.

---

## Phase 10 — Polish

Full pytest suite green, ruff + black clean, `scripts/report.py` prints Phase 8 table from DB.

---

## Composite Score Formula

```python
def composite_score(run, baseline, latency_weight=0.3, memory_weight=0.2):
    accuracy_gain = run.accuracy - baseline.accuracy
    ttft_penalty  = (run.avg_ttft_ms - baseline.avg_ttft_ms) / baseline.avg_ttft_ms
    vram_penalty  = (run.peak_vram_mb - baseline.peak_vram_mb) / baseline.peak_vram_mb
    return accuracy_gain - (latency_weight * ttft_penalty) - (memory_weight * vram_penalty)
```

A config improving TTFT by 25% and VRAM by 40% wins even if MMLU accuracy barely moves.

---

## Resume Claim (fill in real numbers after Phase 8)

> Built an automated LLM optimization platform benchmarking Llama 3.2, executing LoRA fine-tuning on instruction data across a learning rate and epoch sweep, and selecting optimal GGUF quantization configurations — improving MMLU accuracy by X% while reducing VRAM by Y% and TTFT by Z%, served via a custom FastAPI streaming backend.
