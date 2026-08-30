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

MODE = "overfit"  # plumbing -> overfit -> full
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
    #    matched to its GPU (T4, compute 7.5); `-U` pulls a torch wheel with no
    #    kernel image for that card ("no kernel image is available").
    #    peft + trl are pure-python and safe to add without touching torch.
    # trl version must MATCH Kaggle's transformers Trainer signature. transformers
    # >=4.46 renamed the Trainer `tokenizer=` arg to `processing_class=`; trl adopted
    # that in 0.12. Kaggle ships >=4.46 (0.11.4 crashed: "unexpected keyword
    # 'tokenizer'"), so we need trl >=0.12 — but past the buggy chunked-CE patch in
    # 0.12.0/.1 that crashed init ('functools.partial' has no '__func__'). 0.13.0
    # clears both. --no-deps so we don't drag torch off the T4-matched build.
    # bitsandbytes is the CUDA lib that does the 4-bit (nf4) quantization for the
    # QLoRA rungs (overfit/full). Kaggle's image lacks an importable one; transformers
    # 5 requires >=0.46.1. It ships as a self-contained CUDA wheel, so --no-deps
    # installs it without dragging torch off the T4-matched build.
    sh("pip -q install --no-deps 'peft>=0.12,<0.14' 'trl==0.13.0' 'bitsandbytes>=0.46.1'")

    # Print the base versions we're pairing against, so any remaining mismatch names
    # the exact library to target next (no more guessing).
    sh("python -c \"import transformers,accelerate,trl,peft,bitsandbytes;"
       "print('VERSIONS transformers',transformers.__version__,"
       "'accelerate',accelerate.__version__,'trl',trl.__version__,'peft',peft.__version__,"
       "'bnb',bitsandbytes.__version__)\"")

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
