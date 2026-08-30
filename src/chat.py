"""
Manual chat REPL — the "vibe check" the eval numbers can't give you.

Load the model (base, or base + your fine-tuned adapter) and talk to it.
Type a request; watch the raw tool-call it emits. Do this BEFORE fine-tuning
(watch plain Qwen fumble) and AFTER (watch it emit clean calls), with your OWN
made-up prompts — that tests generalization beyond the eval distribution.

    # before FT — plain base model
    python src/chat.py --base

    # after FT — base + your trained adapter
    python src/chat.py --adapter outputs/adapter

Also has a non-interactive mode for the headless kernel: --prompts runs a fixed
list so you get output even when there's no TTY.
"""

from __future__ import annotations

import argparse

import torch

FULL_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# a couple of tools the model can "see" — mimics the glaive system format
DEFAULT_TOOLS = """You are a helpful assistant with access to the following functions. Use them if required -
{"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
{"name": "get_stock_price", "description": "Get current stock price", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}"""

CANNED_PROMPTS = [
    "What's the weather in Tokyo right now?",
    "How much is Tesla stock trading at?",
    "Tell me a joke.",  # no tool applies — should NOT hallucinate a call
]


def load(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # P100 (Pascal) has no bf16
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(FULL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        FULL_MODEL, quantization_config=quant, dtype=torch.float16, device_map="auto"
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"[loaded base + adapter: {args.adapter}]")
    else:
        print("[loaded BASE model, no adapter]")
    return model, tok


def generate(model, tok, user_msg: str) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_TOOLS},
        {"role": "user", "content": user_msg},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    # only decode the newly generated tokens
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true", help="use the plain base model")
    ap.add_argument("--adapter", default=None, help="path to a trained LoRA adapter")
    ap.add_argument("--prompts", action="store_true", help="run canned prompts (headless)")
    args = ap.parse_args()

    model, tok = load(args)

    if args.prompts:
        for p in CANNED_PROMPTS:
            print("\nYOU:", p)
            print("MODEL:", generate(model, tok, p))
        return

    print("\nType a request (Ctrl-C to quit). The model sees get_weather + get_stock_price.\n")
    try:
        while True:
            user = input("YOU: ").strip()
            if not user:
                continue
            print("MODEL:", generate(model, tok, user), "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
