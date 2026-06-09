from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.storage.db import init_db

_startup_time = time.time()
_active_model: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _active_model

    init_db()

    base_model = os.environ.get("BASE_MODEL_ID")
    if base_model:
        from src.inference.hf_engine import ModelRegistry

        name = base_model.split("/")[-1].lower()
        ModelRegistry.load(name, base_model, load_in_4bit=True)
        _active_model = name

    yield

    from src.inference.hf_engine import ModelRegistry

    ModelRegistry.unload_all()


app = FastAPI(title="llm-bench", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.api.routes import router  # noqa: E402

app.include_router(router)


def get_startup_time() -> float:
    return _startup_time


def get_active_model() -> str | None:
    return _active_model
