"""
Kaggle kernel entry point — headless. This is what `kaggle kernels push` runs.

It clones the public repo, installs the few libs Kaggle's image lacks, then
climbs the staircase for the requested MODE. Each rung is cheap and gates the
next, so failures surface in minutes, not hours.

MODE is set below; change it and re-push to advance a rung:
    plumbing -> overfit -> full

Nothing here needs an HF token (HF push is deferred). The adapter is written to
/kaggle/working/outputs so it comes back via `kaggle kernels output`.
"""

import os
import subprocess
import sys

MODE = "plumbing"  # plumbing -> overfit -> full
REPO = "https://github.com/balaguhanesh/tool-call-ft.git"
WORK = "/kaggle/working/tool-call-ft"


def sh(cmd: str):
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    # 1. get the code
    if not os.path.isdir(WORK):
        sh(f"git clone --depth 1 {REPO} {WORK}")
    os.chdir(WORK)

    # 2. Install ONLY the libs Kaggle's image lacks. Do NOT upgrade torch /
    #    bitsandbytes / transformers / accelerate: Kaggle ships a torch build
    #    matched to its GPU (P100, compute 6.0), and `-U` pulls a torch wheel
    #    with no kernel image for that card ("no kernel image is available").
    #    peft + trl are pure-python and safe to add without touching torch.
    sh("pip -q install --no-deps 'peft>=0.12' 'trl>=0.9'")

    # 3. build the data splits (from HF), then assert them BEFORE training
    n_train = 2000 if MODE == "full" else 60
    sh(f"python src/data_prep.py --n-train {n_train} --n-eval 300")
    sh("python src/check_data.py --path data/train.jsonl")
    sh("python src/check_data.py --path data/eval.jsonl")

    # 4. climb the requested rung
    sh(f"python src/train.py --mode {MODE} --data-dir data --out-dir /kaggle/working/outputs/adapter")

    print(f"\n=== rung '{MODE}' complete ===", flush=True)


if __name__ == "__main__":
    main()
