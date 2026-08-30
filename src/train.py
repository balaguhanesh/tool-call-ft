"""
QLoRA fine-tune of Qwen2.5-3B-Instruct for tool-calling — the staircase.

Runs on Kaggle GPU (headless kernel). Three modes, cheapest first, so we catch
failures in minutes on the laptop/GPU instead of hours into a real run:

    --mode plumbing   tiny stand-in model, 5 steps      (~1-2 min)  code path works?
    --mode overfit    Qwen-3B, 10 examples, ~60 steps   (~3-5 min)  loss -> ~0? learning works?
    --mode full       Qwen-3B, real 2k examples, 1 epoch (~1.5-2.5 hr) the actual run

Rule: never run --mode full until plumbing and overfit are both green.

The learning loop itself is trl.SFTTrainer — supervised (imitation) fine-tuning.
We never write the forward pass, loss, or backprop; we configure them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


# tiny stand-in for the plumbing rung — loads in seconds, tests the code path only
PLUMBING_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FULL_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def load_splits(data_dir: str):
    from datasets import load_dataset

    files = {"train": f"{data_dir}/train.jsonl", "eval": f"{data_dir}/eval.jsonl"}
    return load_dataset("json", data_files=files)


def build_model(model_id: str, use_4bit: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Kaggle's GPU is a P100 (Pascal, compute 6.0) — bf16 is unsupported there,
    # so use fp16 for compute + model dtype.
    quant = None
    if use_4bit:
        # QLoRA: base weights frozen in 4-bit; only LoRA adapters train.
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    # Model dtype: for the 4-bit rungs the base is quantized and LoRA adapters stay
    # fp32, so mixed-precision fp16 (via SFTConfig fp16=True) has an fp32 master to
    # unscale grads into. But the plumbing rung trains the RAW model (no 4-bit, no
    # LoRA) — if we also load it in fp16, the params ARE fp16 and the GradScaler
    # can't unscale them ("Attempting to unscale FP16 gradients"). So load the raw
    # plumbing model in fp32 and let fp16=True do the mixed-precision autocast.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant,
        dtype=torch.float16 if use_4bit else torch.float32,
        device_map="auto",
    )
    return model, tok


def attach_lora(model):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # attention + MLP projections — where tool-call behavior lives
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()  # sanity: should be ~0.5% of total
    return model


def to_text(example, tok):
    """Render one record's chat messages via Qwen's chat template into a training string."""
    return tok.apply_chat_template(example["messages"], tokenize=False)


def train(args):
    from trl import SFTConfig, SFTTrainer

    plumbing = args.mode == "plumbing"
    model_id = PLUMBING_MODEL if plumbing else FULL_MODEL

    ds = load_splits(args.data_dir)
    if args.mode == "overfit":
        ds["train"] = ds["train"].select(range(10))  # memorize 10 -> loss should crater
    elif plumbing:
        ds["train"] = ds["train"].select(range(min(16, len(ds["train"]))))

    model, tok = build_model(model_id, use_4bit=not plumbing)
    if not plumbing:
        model = attach_lora(model)

    # step budget per rung
    if plumbing:
        max_steps, epochs = 5, None
    elif args.mode == "overfit":
        max_steps, epochs = 60, None
    else:
        max_steps, epochs = None, 1

    cfg = SFTConfig(
        output_dir=args.out_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        max_steps=max_steps if max_steps else -1,
        num_train_epochs=epochs if epochs else 1,
        fp16=True,  # P100 (Pascal) supports fp16, not bf16
        save_strategy="epoch" if not plumbing else "no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        args=cfg,
        formatting_func=lambda ex: to_text(ex, tok),
    )
    trainer.train()

    if not plumbing:
        trainer.save_model(args.out_dir)  # saves the LoRA adapter only (~50MB)
        tok.save_pretrained(args.out_dir)
        print(f"adapter saved -> {args.out_dir}")

    # loud check for the overfit rung: did loss actually crater?
    if args.mode == "overfit":
        last = trainer.state.log_history[-1].get("loss")
        print(f"OVERFIT final loss = {last}")
        if last is not None and last > 0.5:
            print("WARNING: loss did not crater on 10 examples — learning may be broken.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plumbing", "overfit", "full"], required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="outputs/adapter")
    args = ap.parse_args()
    train(args)
