"""
Data assertions — the cheap laptop-side gate before any GPU run.

The most expensive bug in fine-tuning is the SILENT one: mislabeled or empty
targets that let training "succeed" and produce a useless model, discovered
only at eval hours later. These assertions convert those silent failures into
loud, instant ones. Run this after data_prep, before pushing to Kaggle.

    uv run python src/check_data.py --path data/train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check(path: str) -> None:
    p = Path(path)
    assert p.exists(), f"missing file: {p}"

    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    assert rows, f"{p} is EMPTY — data_prep produced nothing (parser bug?)"

    n_bad = 0
    for i, r in enumerate(rows):
        # shape
        assert "messages" in r, f"row {i}: no 'messages'"
        assert "gold_name" in r and "gold_args" in r, f"row {i}: missing gold fields"

        msgs = r["messages"]
        roles = [m["role"] for m in msgs]
        assert "assistant" in roles, f"row {i}: no assistant target to learn from"

        # the assistant target must be non-empty and parse as a tool call
        asst = next(m["content"] for m in msgs if m["role"] == "assistant")
        assert asst.strip(), f"row {i}: EMPTY assistant target (nothing to imitate)"
        try:
            call = json.loads(asst)
        except json.JSONDecodeError:
            n_bad += 1
            continue
        assert call.get("name") == r["gold_name"], f"row {i}: assistant/gold name mismatch"

        # gold_name must be a non-empty identifier-ish string
        assert isinstance(r["gold_name"], str) and r["gold_name"], f"row {i}: bad gold_name"
        assert isinstance(r["gold_args"], dict), f"row {i}: gold_args not a dict"

    frac_bad = n_bad / len(rows)
    assert frac_bad < 0.02, f"{frac_bad:.1%} of assistant targets don't parse — pipeline is broken"

    print(f"OK  {p.name}: {len(rows)} rows, {n_bad} unparseable assistant targets ({frac_bad:.2%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/train.jsonl")
    args = ap.parse_args()
    check(args.path)
