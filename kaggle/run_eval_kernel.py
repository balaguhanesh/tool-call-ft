"""
Kaggle kernel entry point for the base-vs-tuned EVAL (headless).

Separate from run_kernel.py (which trains). This one clones the repo, installs
the same libs, rebuilds the held-out eval split, then runs run_eval.py which
loads Qwen2.5-3B twice (base, then base+adapter) and prints the before/after
table.

The trained adapter comes from the training kernel, mounted as a kernel source
at /kaggle/input/tool-call-ft/outputs/adapter (set in kernel-metadata via
kernel_sources). No re-download, no HF push needed.
"""

import os
import subprocess

REPO = "https://github.com/balaguhanesh/tool-call-ft.git"
WORK = "/kaggle/working/tool-call-ft"
ADAPTER = "/kaggle/input/tool-call-ft/outputs/adapter"


def sh(cmd: str):
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    if not os.path.isdir(WORK):
        sh(f"git clone --depth 1 {REPO} {WORK}")
    os.chdir(WORK)

    # same base libs as training (see run_kernel.py for the why of each pin)
    sh("pip -q install --no-deps 'peft>=0.12,<0.14' 'trl==0.13.0' 'bitsandbytes>=0.46.1'")

    # rebuild the SAME held-out eval split the model never trained on
    sh("python src/data_prep.py --n-train 60 --n-eval 300")
    sh("python src/check_data.py --path data/eval.jsonl")

    # locate the adapter (mounted kernel source); fail loudly if it's not there
    if not os.path.isdir(ADAPTER):
        raise SystemExit(f"adapter not found at {ADAPTER} — check kernel_sources mount")

    sh(f"python src/run_eval.py --eval data/eval.jsonl --adapter {ADAPTER}")

    print("\n=== eval complete ===", flush=True)


if __name__ == "__main__":
    main()
