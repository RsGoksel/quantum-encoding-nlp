"""
Multi-seed batch runner for encoding experiments.
Runs 4 encodings x N seeds on a given dataset, sequentially.

Usage:
    PYTHONIOENCODING=utf-8 python experiments/run_multiseed.py --dataset imdb --seeds 42 123 456 789 2024
    PYTHONIOENCODING=utf-8 python experiments/run_multiseed.py --dataset sst2 --seeds 42 --include_baselines
"""
import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
PROJECT_ROOT = Path(__file__).parent.parent

ENCODINGS = ["enc_angle", "enc_dense", "enc_iqp", "enc_reupload"]
BASELINES = ["baseline", "attention"]


def check_exists(model, dataset, seed):
    """Check if experiment results already exist."""
    suffix = f"_seed{seed}"
    history_path = PROJECT_ROOT / "results" / "logs" / f"{model}_{dataset}{suffix}_history.json"
    return history_path.exists()


def run_experiment(model, dataset, seed, reps=1, lr_quantum=0.05, skip_existing=True):
    """Run a single training experiment via subprocess."""
    if skip_existing and check_exists(model, dataset, seed):
        print(f"  [SKIP] {model} seed={seed} (already exists)")
        return True, 0.0

    suffix = f"_seed{seed}"
    cmd = [
        sys.executable, str(PROJECT_ROOT / "training" / "train.py"),
        "--model", model,
        "--dataset", dataset,
        "--seed", str(seed),
        "--reps", str(reps),
        "--lr_quantum", str(lr_quantum),
        "--subset",
        "--epochs", "30",
        "--suffix", suffix,
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0

    success = result.returncode == 0
    status = "OK" if success else "FAIL"
    print(f"  [{status}] {model} seed={seed} ({elapsed:.0f}s)")
    return success, elapsed


def main():
    parser = argparse.ArgumentParser(description="Multi-seed experiment runner")
    parser.add_argument("--dataset", default="imdb", choices=["imdb", "sst2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2024])
    parser.add_argument("--include_baselines", action="store_true")
    args = parser.parse_args()

    models = list(ENCODINGS)
    if args.include_baselines:
        models = BASELINES + models

    total = len(models) * len(args.seeds)
    print(f"{'='*60}")
    print(f"Multi-seed Experiment Runner")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Models: {models}")
    print(f"Seeds: {args.seeds}")
    print(f"Total experiments: {total}")
    print(f"{'='*60}")

    done = 0
    failed = 0
    total_time = 0

    for seed in args.seeds:
        for model in models:
            done += 1
            print(f"\n[{done}/{total}] {model} | {args.dataset} | seed={seed}")
            success, elapsed = run_experiment(model, args.dataset, seed)
            total_time += elapsed
            if not success:
                failed += 1

    print(f"\n{'='*60}")
    print(f"DONE: {done - failed}/{total} succeeded, {failed} failed")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
