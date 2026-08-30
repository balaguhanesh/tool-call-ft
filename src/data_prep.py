"""
Data prep: glaive-function-calling-v2 -> chat-formatted train/eval splits.

This file defines the SHAPE of the problem:
  - what the model sees (the prompt: user request + available tools)
  - what counts as the gold answer (the tool call: name + arguments)

Everything downstream — the training format AND your grader in eval.py —
depends on the shapes produced here. Run this locally (no GPU) to eyeball
the output before we ever touch Kaggle.

Source dataset: glaiveai/glaive-function-calling-v2 (Hugging Face Hub)
Each raw row is a multi-turn 'chat' string with embedded system tools and
assistant turns that sometimes contain a <functioncall> {...} span.

Output: two JSONL files, train.jsonl / eval.jsonl, each line:
  {
    "messages": [ {role, content}, ... ],   # chat-templated at train time
    "gold_name": "get_weather",             # for eval grading
    "gold_args": {"city": "Paris"}          # for eval grading
  }
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# glaive encodes tool calls inline like:
#   <functioncall> {"name": "get_x", "arguments": '{"a": 1, "b": true}'}
# Note the SINGLE quotes around the arguments value — that is NOT valid JSON,
# so we can't json.loads the whole span. We pull name + args-string separately.
FUNCTIONCALL_RE = re.compile(r"<functioncall>\s*(\{.*?\})\s*(?:<\|endoftext\|>|$)", re.DOTALL)
_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
# arguments value is wrapped in single quotes: 'arguments': '{...}'
_ARGS_RE = re.compile(r'"arguments"\s*:\s*\'(\{.*?\})\'', re.DOTALL)


def parse_functioncall(assistant_text: str) -> tuple[str, dict] | None:
    """
    Pull (name, args_dict) out of an assistant turn that contains a tool call.

    glaive's format is awkward: the outer object looks like JSON but the
    'arguments' value is a SINGLE-quoted JSON string, so the whole span is not
    valid JSON. We extract name and the args-string with regexes, then json.loads
    only the (now unquoted) args string. Returns None if this isn't a tool call.
    """
    if "<functioncall>" not in assistant_text:
        return None
    m = FUNCTIONCALL_RE.search(assistant_text)
    if not m:
        return None
    span = m.group(1)
    name_m = _NAME_RE.search(span)
    if not name_m:
        return None
    name = name_m.group(1)
    args_m = _ARGS_RE.search(span)
    args: dict = {}
    if args_m:
        try:
            args = json.loads(args_m.group(1))
        except json.JSONDecodeError:
            args = {}
    return name, args


def build_example(system: str, user: str, gold_name: str, gold_args: dict) -> dict:
    """
    Assemble one training/eval record.

    The 'messages' become the training target via the chat template. We keep
    gold_name / gold_args as flat fields so eval.py can grade without re-parsing.
    The assistant's gold content is the canonical JSON tool call.
    """
    gold_call = json.dumps({"name": gold_name, "arguments": gold_args})
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": gold_call},
        ],
        "gold_name": gold_name,
        "gold_args": gold_args,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset  # imported here so the file loads without datasets installed

    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    ds = ds.shuffle(seed=args.seed)

    records: list[dict] = []
    for row in ds:
        # NOTE: exact field names/structure of glaive rows are inspected at first run;
        # this loop is the piece we'll harden once we see one real row (see __main__ probe).
        system = row.get("system", "")
        chat = row.get("chat", "")
        # naive split of the multi-turn chat string into user / assistant turns
        turns = re.split(r"(USER:|ASSISTANT:)", chat)
        user_text, parsed = None, None
        for i in range(1, len(turns) - 1, 2):
            tag, body = turns[i], turns[i + 1].strip()
            if tag == "USER:":
                user_text = body
            elif tag == "ASSISTANT:" and user_text:
                parsed = parse_functioncall(body)
                if parsed:
                    records.append(build_example(system, user_text, parsed[0], parsed[1]))
                    break
        if len(records) >= args.n_train + args.n_eval:
            break

    train, ev = records[: args.n_train], records[args.n_train : args.n_train + args.n_eval]
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    for name, split in [("train", train), ("eval", ev)]:
        with open(out / f"{name}.jsonl", "w") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(split):5d} -> {out / f'{name}.jsonl'}")


if __name__ == "__main__":
    main()
