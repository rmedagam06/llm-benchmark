from __future__ import annotations

import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

router = APIRouter()


class InferRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    stream: bool = False


# ---------------------------------------------------------------------------
# /infer
# ---------------------------------------------------------------------------


@router.post("/infer")
async def infer(req: InferRequest):
    from src.inference.hf_engine import ModelRegistry

    engine = ModelRegistry.get(req.model)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not loaded")

    if req.stream:
        return EventSourceResponse(_stream_tokens(engine, req))

    text, metrics = engine.generate_with_metrics(
        req.prompt, max_new_tokens=req.max_tokens, temperature=req.temperature
    )
    return {"text": text, "metrics": metrics.to_dict()}


async def _stream_tokens(engine, req: InferRequest) -> AsyncGenerator[dict, None]:
    tokens: list[str] = []
    first = True
    for tok in engine.generate(
        req.prompt, max_new_tokens=req.max_tokens, temperature=req.temperature, stream=True
    ):
        if first:
            first = False
        tokens.append(tok)
        yield {"data": json.dumps({"token": tok, "done": False})}

    text = "".join(tokens)
    total_toks = len(engine.tokenizer.encode(text))
    yield {
        "data": json.dumps(
            {
                "token": "",
                "done": True,
                "metrics": {
                    "model_name": engine.model_id,
                    "total_tokens": total_toks,
                },
            }
        )
    }


# ---------------------------------------------------------------------------
# /models
# ---------------------------------------------------------------------------


@router.get("/models")
def list_models():
    from pathlib import Path

    from src.inference.hf_engine import ModelRegistry

    loaded = list(ModelRegistry._engines.keys())

    gguf_files = []
    models_dir = Path("./models/quantized")
    if models_dir.exists():
        gguf_files = [p.name for p in models_dir.glob("*.gguf")]

    return {"loaded_hf": loaded, "gguf_files": gguf_files}


# ---------------------------------------------------------------------------
# /results
# ---------------------------------------------------------------------------


@router.get("/results")
def get_results(model_name: str | None = None):
    from src.storage.db import get_all_benchmarks

    rows = get_all_benchmarks(model_name=model_name)
    return [
        {
            "id": r.id,
            "model_name": r.model_name,
            "adapter_name": r.adapter_name,
            "quant_level": r.quant_level,
            "accuracy": r.accuracy,
            "avg_ttft_ms": r.avg_ttft_ms,
            "peak_vram_mb": r.peak_vram_mb,
            "num_questions": r.num_questions,
            "timestamp": r.timestamp,
        }
        for r in rows
    ]


@router.get("/results/compare")
def compare_results(baseline_id: int, candidate_id: int):
    from src.storage.db import compare

    delta = compare(baseline_id, candidate_id)
    if not delta:
        raise HTTPException(status_code=404, detail="One or both run IDs not found")
    return delta


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@router.get("/health")
def health():
    from src.api.main import get_active_model, get_startup_time
    from src.monitoring.metrics import VRAMMonitor

    vram = VRAMMonitor()
    vram.start()
    import time as _time

    _time.sleep(0.15)
    vram.stop()

    return {
        "status": "ok",
        "active_model": get_active_model(),
        "vram_mb": round(vram.peak_vram_mb, 1),
        "uptime_s": round(time.time() - get_startup_time(), 1),
    }
