from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


@dataclass
class LoRAConfig:
    base_model_id: str
    dataset_name: str = "teknium/OpenHermes-2.5"
    max_samples: int = 3000
    r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    learning_rate: float = 2e-4
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    output_dir: str = "./adapters"
    max_seq_length: int = 512
    seed: int = 42


def _format_sample(sample: dict) -> str:
    instruction = sample.get("instruction") or sample.get("conversations", [{}])[0].get("value", "")
    output = sample.get("output") or sample.get("conversations", [{}])[-1].get("value", "")
    return f"<|user|>\n{instruction}\n<|assistant|>\n{output}"


def train_lora(config: LoRAConfig):
    """Run QLoRA fine-tuning and return TrainingMetrics."""
    from datasets import load_dataset
    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    from src.monitoring.metrics import TrainingMetrics, VRAMMonitor

    # Reproducibility
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = PeftLoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)

    # Load dataset slice — hard cap prevents downloading full 1M-row dataset
    ds = load_dataset(
        config.dataset_name, split=f"train[:{config.max_samples}]", trust_remote_code=True
    )

    def formatting_fn(batch):
        return [_format_sample(s) for s in batch]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    model_slug = config.base_model_id.split("/")[-1].lower()
    lr_str = f"{config.learning_rate:.0e}".replace("-0", "-").replace("+0", "")
    adapter_dir = Path(config.output_dir) / f"{model_slug}-lr{lr_str}-ep{config.num_epochs}-{ts}"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=config.seed,
        report_to="none",
    )

    vram = VRAMMonitor()
    vram.start()
    t_start = time.perf_counter()

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        formatting_func=formatting_fn,
        args=training_args,
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
    )
    trainer.train()

    training_time = time.perf_counter() - t_start
    vram.stop()

    trainer.save_model(str(adapter_dir))

    final_loss = trainer.state.log_history[-1].get("train_loss", 0.0)

    from src.storage.db import save_training

    metrics = TrainingMetrics(
        model_name=config.base_model_id,
        adapter_name=str(adapter_dir),
        learning_rate=config.learning_rate,
        epochs=config.num_epochs,
        training_time_seconds=training_time,
        final_loss=final_loss,
        peak_vram_mb=vram.peak_vram_mb,
        seed=config.seed,
    )

    try:
        save_training(metrics)
    except Exception:
        pass

    return metrics, str(adapter_dir)
