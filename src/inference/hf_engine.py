from __future__ import annotations

import time
from threading import Thread
from typing import Generator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from src.monitoring.metrics import InferenceMetrics, TTFTTimer, VRAMMonitor


class HFEngine:
    """HuggingFace transformers inference engine with optional 4-bit quantization and LoRA support."""

    def __init__(
        self,
        model_id: str,
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ) -> None:
        self.model_id = model_id
        self._adapter_name: str | None = None

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map if load_in_4bit else None,
            dtype=torch.float32 if not load_in_4bit else None,
        )
        if not load_in_4bit and device_map == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)

        self.model.eval()

    # ------------------------------------------------------------------
    def load_adapter(self, adapter_path: str) -> None:
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self._adapter_name = adapter_path

    def unload_adapter(self) -> None:
        if hasattr(self.model, "disable_adapter"):
            self.model.disable_adapter()
        self._adapter_name = None

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if stream:
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                streamer=streamer,
            )
            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            for token in streamer:
                yield token
            thread.join()
        else:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )
            input_len = inputs["input_ids"].shape[1]
            yield self.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

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
        total_tokens = len(self.tokenizer.encode(text))
        tps = total_tokens / elapsed if elapsed > 0 else 0.0

        metrics = InferenceMetrics(
            model_name=self.model_id,
            adapter_name=self._adapter_name,
            ttft_ms=ttft.ttft_ms,
            tokens_per_second=tps,
            peak_vram_mb=vram.peak_vram_mb,
            total_tokens=total_tokens,
        )
        return text, metrics

    # ------------------------------------------------------------------
    def unload(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------


class ModelRegistry:
    """Singleton registry mapping names to loaded HFEngine instances."""

    _engines: dict[str, HFEngine] = {}

    @classmethod
    def load(cls, name: str, model_id: str, **kwargs) -> HFEngine:
        if name not in cls._engines:
            cls._engines[name] = HFEngine(model_id, **kwargs)
        return cls._engines[name]

    @classmethod
    def get(cls, name: str) -> HFEngine | None:
        return cls._engines.get(name)

    @classmethod
    def unload_all(cls) -> None:
        for engine in cls._engines.values():
            engine.unload()
        cls._engines.clear()
