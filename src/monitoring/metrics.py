from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


class VRAMMonitor:
    """Samples VRAM (or RSS on CPU-only) every 100 ms in a background thread."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.1):
        self._device_index = device_index
        self._interval_s = interval_s
        self._samples: list[float] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    def _sample(self) -> float:
        if _NVML_AVAILABLE:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / 1024 / 1024  # bytes → MB
        return psutil.Process().memory_info().rss / 1024 / 1024

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._samples.append(self._sample())
            time.sleep(self._interval_s)

    # ------------------------------------------------------------------
    def start(self) -> "VRAMMonitor":
        self._samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def __enter__(self) -> "VRAMMonitor":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    @property
    def peak_vram_mb(self) -> float:
        return max(self._samples, default=0.0)

    @property
    def avg_vram_mb(self) -> float:
        return sum(self._samples) / len(self._samples) if self._samples else 0.0


# ---------------------------------------------------------------------------


class TTFTTimer:
    """Measures time-to-first-token in milliseconds."""

    def __init__(self) -> None:
        self._t_request: float | None = None
        self._t_first_token: float | None = None

    def mark_request(self) -> None:
        self._t_request = time.perf_counter()

    def mark_first_token(self) -> None:
        self._t_first_token = time.perf_counter()

    @property
    def ttft_ms(self) -> float:
        if self._t_request is None or self._t_first_token is None:
            return 0.0
        return (self._t_first_token - self._t_request) * 1000.0


# ---------------------------------------------------------------------------


@dataclass
class InferenceMetrics:
    model_name: str
    adapter_name: str | None
    ttft_ms: float
    tokens_per_second: float
    peak_vram_mb: float
    total_tokens: int

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "adapter_name": self.adapter_name,
            "ttft_ms": self.ttft_ms,
            "tokens_per_second": self.tokens_per_second,
            "peak_vram_mb": self.peak_vram_mb,
            "total_tokens": self.total_tokens,
        }


@dataclass
class TrainingMetrics:
    model_name: str
    adapter_name: str
    learning_rate: float
    epochs: int
    training_time_seconds: float
    final_loss: float
    peak_vram_mb: float
    seed: int

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "adapter_name": self.adapter_name,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "training_time_seconds": self.training_time_seconds,
            "final_loss": self.final_loss,
            "peak_vram_mb": self.peak_vram_mb,
            "seed": self.seed,
        }
