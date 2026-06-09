from __future__ import annotations

import time
from typing import Generator

from src.monitoring.metrics import InferenceMetrics, TTFTTimer, VRAMMonitor


class GGUFEngine:
    """Inference engine backed by llama-cpp-python. Same interface as HFEngine."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1) -> None:
        from llama_cpp import Llama

        self.model_path = model_path
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        self._adapter_name: str | None = None

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        if stream:
            for chunk in self._llm(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                stream=True,
            ):
                yield chunk["choices"][0]["text"]
        else:
            result = self._llm(prompt, max_tokens=max_new_tokens, temperature=temperature)
            yield result["choices"][0]["text"]

    # ------------------------------------------------------------------
    def generate_with_metrics(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> tuple[str, InferenceMetrics]:
        vram = VRAMMonitor()
        ttft = TTFTTimer()
        tokens: list[str] = []

        vram.start()
        ttft.mark_request()
        t_start = time.perf_counter()

        first = True
        for tok in self.generate(prompt, max_new_tokens, temperature, stream=True):
            if first:
                ttft.mark_first_token()
                first = False
            tokens.append(tok)

        elapsed = time.perf_counter() - t_start
        vram.stop()

        text = "".join(tokens)
        total_tokens = len(text.split())
        tps = total_tokens / elapsed if elapsed > 0 else 0.0

        import os

        metrics = InferenceMetrics(
            model_name=os.path.basename(self.model_path),
            adapter_name=None,
            ttft_ms=ttft.ttft_ms,
            tokens_per_second=tps,
            peak_vram_mb=vram.peak_vram_mb,
            total_tokens=total_tokens,
        )
        return text, metrics

    def unload(self) -> None:
        del self._llm
