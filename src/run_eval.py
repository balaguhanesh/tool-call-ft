"""
Base-vs-tuned eval runner — the proof-of-work payoff.

Loads the held-out eval.jsonl, runs Qwen2.5-3B TWICE over it — once as the raw
base model, once with the trained LoRA adapter attached — and prints the two
metric rows (json_valid / name_acc / arg_match) side by side. The only thing
that changes between the two runs is the adapter, so the delta IS the effect of
fine-tuning.

Generation is GREEDY (do_sample=False): deterministic and reproducible, so the
before/after numbers aren't polluted by sampling noise. The model sees only the
system+user turns (chat-templated with a generation prompt); its generated
assistant text is what we grade.

Usage (on Kaggle T4):
    python src/run_eval.py --eval data/eval.jsonl --adapter /kaggle/input/.../adapter
"""

from __future__ import annotations

import argparse
import json

import torch

from eval import Example, evaluate
from train import FULL_MODEL, build_model

MAX_NEW_TOKENS = 128  # tool calls are short; enough for name + args JSON


def load_eval_examples(path: str):
    """eval.jsonl -> (list[Example], list[messages]). The model input is the
    system+user turns; the assistant turn is the gold target we don't feed in."""
    examples, prompt_msgs = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = rec["messages"]
            # keep everything up to (but not including) the assistant answer
            ctx = [m for m in msgs if m["role"] in ("system", "user")]
            examples.append(
                Example(
                    prompt="",  # unused: we drive generation via the messages below
                    gold_name=rec["gold_name"],
                    gold_args=rec["gold_args"],
                )
            )
            prompt_msgs.append(ctx)
    return examples, prompt_msgs


def make_model_fn(model, tok, prompt_msgs):
    """Return a model_fn(prompt)->text. We ignore the string prompt and index into
    prompt_msgs by call order, because evaluate() iterates examples in order and we
    need the full chat-message structure, not a flat string."""
    call = {"i": 0}

    def model_fn(_prompt: str) -> str:
        msgs = prompt_msgs[call["i"]]
        call["i"] += 1
        # transformers 5 returns a BatchEncoding (input_ids + attention_mask), not a
        # bare tensor — pass return_dict=True and unpack with ** so generate() gets
        # both the ids and the mask (mask also silences the pad-token warning).
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # greedy: deterministic before/after
                pad_token_id=tok.pad_token_id,
            )
        gen = out[0][prompt_len:]  # only the newly generated tokens
        return tok.decode(gen, skip_special_tokens=True)

    return model_fn


def run_one(label, adapter_path, eval_path):
    from transformers import AutoTokenizer

    examples, prompt_msgs = load_eval_examples(eval_path)
    model, tok = build_model(FULL_MODEL, use_4bit=True)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    report = evaluate(make_model_fn(model, tok, prompt_msgs), examples)
    print(f"RESULT {label}: {json.dumps(report)}")
    # free the GPU before the next model loads
    del model
    torch.cuda.empty_cache()
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval.jsonl")
    ap.add_argument("--adapter", required=True, help="path to trained LoRA adapter")
    args = ap.parse_args()

    base = run_one("base", None, args.eval)
    tuned = run_one("tuned", args.adapter, args.eval)

    # the before/after table — the whole point
    print("\n=== BASE vs TUNED (held-out eval) ===")
    print(f"{'metric':<12}{'base':>10}{'tuned':>10}{'delta':>10}")
    for k in ("json_valid", "name_acc", "arg_match"):
        d = round(tuned[k] - base[k], 4)
        print(f"{k:<12}{base[k]:>10}{tuned[k]:>10}{d:>+10}")
    print(f"n = {base['n']}")


if __name__ == "__main__":
    main()
