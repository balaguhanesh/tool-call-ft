# tool-call-ft

QLoRA fine-tune of **Qwen2.5-3B-Instruct** for reliable function / tool calling.

Proof-of-work: teach a small open model to emit correct tool calls, and prove
it worked with an objective before/after eval — not vibes.

## The claim (fill in after training)

| Metric | Base | Fine-tuned |
|---|---|---|
| JSON-valid rate | – | – |
| Function-name accuracy | – | – |
| Argument match | – | – |

## Stack

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Method: QLoRA (4-bit base via `bitsandbytes`, LoRA adapters via `peft`)
- Trainer: `trl.SFTTrainer`
- Data: `glaiveai/glaive-function-calling-v2` (Hugging Face Hub)
- Compute: Kaggle free P100 (16GB)
- Result: LoRA adapter pushed to the Hugging Face Hub

## Layout

```
src/
  eval.py        # eval harness — grader core is the intellectual work
  data_prep.py   # dataset -> chat-formatted train/eval splits
notebooks/
  train.ipynb    # the Kaggle training notebook (built after the harness)
```

## How the eval is graded

For each held-out example we compare the model's predicted tool call against the
gold tool call on three axes, independently:

1. **JSON-valid** — does the output parse as JSON at all
2. **Function name** — is the chosen tool correct
3. **Arguments** — do the arguments match

See `src/eval.py`.
