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

import glob
import os
import subprocess

REPO = "https://github.com/balaguhanesh/tool-call-ft.git"
WORK = "/kaggle/working/tool-call-ft"


def find_adapter():
    """Kaggle mounts the source kernel's output under /kaggle/input/<slug>/... but
    the exact path (and whether the 'outputs/' prefix survives) varies. So don't
    hard-code it: find the adapter by its config file anywhere under /kaggle/input
    and return that directory. Prefer the top-level adapter over checkpoint-* dirs."""
    hits = glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
    if not hits:
        return None
    # a training run leaves checkpoint-N/ subdirs too; pick the final adapter,
    # i.e. the shallowest path that is NOT inside a checkpoint-* folder.
    hits = [h for h in hits if "checkpoint-" not in h] or hits
    hits.sort(key=lambda p: p.count("/"))
    return os.path.dirname(hits[0])


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

    # locate the adapter (mounted kernel source); print what IS under /kaggle/input
    # so a miss is debuggable, then fail loudly.
    adapter = find_adapter()
    if adapter is None:
        sh("find /kaggle/input -maxdepth 4 -type d | head -50")
        raise SystemExit("adapter not found under /kaggle/input — check kernel_sources mount")
    print(f"found adapter -> {adapter}", flush=True)

    sh(f"python src/run_eval.py --eval data/eval.jsonl --adapter {adapter}")

    print("\n=== eval complete ===", flush=True)


if __name__ == "__main__":
    main()
