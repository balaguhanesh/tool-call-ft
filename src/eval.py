"""
Eval harness for tool-call fine-tuning.

The whole proof-of-work rests on this file: it turns "the model feels better"
into three hard numbers. Everything here is boilerplate EXCEPT `grade_tool_call`,
which is where the judgment lives — you write that.

Usage (conceptual):
    from eval import evaluate
    report = evaluate(model_fn, eval_examples)
    print(report)   # {"json_valid": .., "name_acc": .., "arg_match": .., "n": ..}

Where:
    model_fn(prompt: str) -> str        # raw model output text
    eval_examples: list[Example]        # held-out gold set
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Example:
    """One held-out eval item."""
    prompt: str                     # user request + available tools (model input)
    gold_name: str                  # correct tool/function name
    gold_args: dict                 # correct arguments


@dataclass
class GradeResult:
    """Per-example grade. Three independent booleans — do NOT collapse them."""
    json_valid: bool
    name_correct: bool
    args_correct: bool


def extract_tool_call(raw: str) -> dict | None:
    """
    Best-effort: pull a JSON tool call out of raw model text.

    Base models wrap JSON in prose, fence it in ```json, or emit trailing text.
    This tries to recover a dict; returns None if nothing parseable is found.
    Returning None is itself a signal (json_valid = False).
    """
    raw = raw.strip()
    # strip a ```json ... ``` fence if present
    if "```" in raw:
        parts = raw.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[len("json"):].strip()
            try:
                obj = json.loads(p)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    # try the whole thing
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # last resort: first {...} span
    start, depth = raw.find("{"), 0
    if start == -1:
        return None
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(raw[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def grade_tool_call(pred: dict | None, gold_name: str, gold_args: dict) -> GradeResult:
    """
    >>> THIS IS YOURS. <<<

    Given the parsed prediction (or None if it didn't parse) and the gold
    tool name + args, decide the three booleans in GradeResult.

    Rulings for this project (the "strict, honest floor" — loosen later only
    with justification):
      - pred is None (unparseable)  -> all three False. If it can't emit JSON,
        it didn't choose a tool or pass args correctly.
      - name match  -> exact, case-sensitive. Tool names are code identifiers.
      - args match  -> exact dict equality (types, keys, values). Brutal but
        unambiguous; "5" != 5, extra/missing key fails the whole example.

    We read the tool call from pred["name"] and pred["arguments"] (the clean
    JSON shape our data_prep trains toward). A malformed-but-parsed dict that
    lacks those keys counts as json_valid but name/args False.
    """
    if pred is None:
        return GradeResult(json_valid=False, name_correct=False, args_correct=False)

    pred_name = pred.get("name")
    pred_args = pred.get("arguments", {})

    name_correct = pred_name == gold_name
    args_correct = pred_args == gold_args

    return GradeResult(json_valid=True, name_correct=name_correct, args_correct=args_correct)


def evaluate(model_fn: Callable[[str], str], examples: list[Example]) -> dict:
    """Run the model over the eval set and aggregate the three metrics."""
    n = len(examples)
    if n == 0:
        raise ValueError("empty eval set")
    import sys, time
    jv = nc = ac = 0
    t0 = time.time()
    for i, ex in enumerate(examples, 1):
        raw = model_fn(ex.prompt)
        pred = extract_tool_call(raw)
        g = grade_tool_call(pred, ex.gold_name, ex.gold_args)
        jv += int(g.json_valid)
        nc += int(g.name_correct)
        ac += int(g.args_correct)
        # progress heartbeat so a long unbatched run isn't a blind wait: every 25
        # examples print done/total + rate + running arg_match, flushed to the log.
        if i % 25 == 0 or i == n:
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  [{i}/{n}] {rate:.1f} ex/s  running arg_match={ac / i:.3f}",
                  flush=True)
    return {
        "n": n,
        "json_valid": round(jv / n, 4),
        "name_acc": round(nc / n, 4),
        "arg_match": round(ac / n, 4),
    }


if __name__ == "__main__":
    # smoke test with a trivial fake model once grade_tool_call is written
    fake = [
        Example(
            prompt="What's the weather in Paris?",
            gold_name="get_weather",
            gold_args={"city": "Paris"},
        )
    ]
    def model_fn(_):
        return '{"name": "get_weather", "arguments": {"city": "Paris"}}'
    print(evaluate(model_fn, fake))
