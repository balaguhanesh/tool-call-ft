# tool-call-ft

QLoRA fine-tune of **Qwen2.5-3B-Instruct** for reliable function / tool calling.

Proof-of-work: teach a small open model to emit correct tool calls, and prove
it worked with an objective before/after eval — not vibes.

## Results

Base vs. fine-tuned on **300 held-out examples** the model never trained on.
Greedy decoding, strict exact-match grader (see below).

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| JSON-valid rate | 51.7% | **100.0%** | +48.3 |
| Function-name accuracy | 1.0% | **100.0%** | +99.0 |
| Argument match (exact) | 0.3% | **95.0%** | +94.7 |

The base model isn't stupid — it wraps calls in prose, uses its own schema, and
explains instead of emitting the compact `{"name": ..., "arguments": {...}}`
contract. Under a strict exact-match grader that scores ~0% on name/args.
Fine-tuning teaches **format compliance and schema discipline**: emit the exact
tool-call contract, every time. That is what tool-calling reliability *is* in
production.

The remaining 5% argument gap is the honest hard part: the grader demands exact
dict equality, so `"5" != 5` and a single extra/missing key fails the whole row.

## Stack

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Method: QLoRA — 4-bit NF4 base via `bitsandbytes`, LoRA adapters via `peft`
  (r=16, α=32, on all attention + MLP projections; ~30M trainable, 0.96% of params)
- Trainer: `trl.SFTTrainer` (supervised / imitation fine-tuning)
- Data: `glaiveai/glaive-function-calling-v2` (Hugging Face Hub) → 2000 train / 300 eval
- Compute: Kaggle free **T4** (16GB), fp16 mixed precision
- Full run: 1 epoch, ~21 min, final train loss ≈ 0.13

Version pins that matter on Kaggle's current image (transformers 5.0, accelerate 1.13):
`trl==0.13.0`, `bitsandbytes>=0.46.1`, installed `--no-deps` so they don't drag
torch off the T4-matched build.

## The staircase (fail cheap before the expensive run)

Never launch the full run until each cheaper rung is green:

1. **plumbing** — tiny 0.5B model, 5 steps. Does the code path connect end-to-end?
2. **overfit** — 3B, 10 examples, 60 steps. Loss must crater (→ ~0.01). Does it *learn*?
3. **full** — 3B, 2000 examples, 1 epoch. The real run.

Each rung answers a different question, and a failure surfaces in minutes instead
of hours into a real run.

## How the eval is graded

For each held-out example we compare the model's predicted tool call against the
gold tool call on three **independent** axes:

1. **JSON-valid** — does the output parse as JSON at all
2. **Function name** — exact, case-sensitive (tool names are code identifiers)
3. **Arguments** — exact dict equality (types, keys, values all count)

An unparseable output scores False on all three. The grader is deliberately the
"strict, honest floor" — loosen it later only with justification. See
`src/eval.py` (the grader core, `grade_tool_call`, is the intellectual work).

## Layout

```
src/
  data_prep.py         # glaive dataset -> clean chat-formatted train/eval splits
  check_data.py        # data assertions (fail fast locally, before any GPU)
  eval.py              # eval harness + strict grader
  run_eval.py          # base-vs-tuned runner (greedy generation)
  train.py             # the staircase: plumbing / overfit / full
  chat.py              # manual before/after REPL for a qualitative vibe check
kaggle/                # training kernel (headless; clones this repo, climbs a rung)
kaggle-eval/           # eval kernel (mounts the trained adapter, runs the table)
```

## Reproduce

Training and eval run as headless Kaggle kernels (free T4). Both clone this repo,
install the pinned libs, rebuild the data splits, and run.

```bash
# train (set MODE in kaggle/run_kernel.py: plumbing -> overfit -> full)
kaggle kernels push -p kaggle/

# eval the trained adapter (mounted from the training kernel's output)
kaggle kernels push -p kaggle-eval/
```
