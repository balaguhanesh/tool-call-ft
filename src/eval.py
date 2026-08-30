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

    Design decisions you have to make here — this is the actual work:
      - `pred` may be None (unparseable). What are the three booleans then?
      - Where do the function name and args live inside `pred`? Real outputs
        look like {"name": "...", "arguments": {...}} but the exact keys vary
        by chat template — decide how forgiving to be.
      - Argument matching: exact dict equality is brutal (units, "5" vs 5,
        key ordering, extra optional keys). Decide how strict. Strict is
        honest but may under-count; loose flatters the model. Pick and justify.
      - Should a missing/extra argument key fail the whole example, or do you
        score partial? For a first pass, all-or-nothing is defensible.

    Return a GradeResult with the three fields set.
    """
    raise NotImplementedError("grade_tool_call: implement the grading logic")


def evaluate(model_fn: Callable[[str], str], examples: list[Example]) -> dict:
    """Run the model over the eval set and aggregate the three metrics."""
    n = len(examples)
    if n == 0:
        raise ValueError("empty eval set")
    jv = nc = ac = 0
    for ex in examples:
        raw = model_fn(ex.prompt)
        pred = extract_tool_call(raw)
        g = grade_tool_call(pred, ex.gold_name, ex.gold_args)
        jv += int(g.json_valid)
        nc += int(g.name_correct)
        ac += int(g.args_correct)
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
