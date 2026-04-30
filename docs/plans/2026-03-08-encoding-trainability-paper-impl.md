# Encoding Trainability Paper — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run multi-seed and cross-dataset experiments, generate paper figures, and produce all data needed for a short conference paper on quantum encoding selection via gradient trainability.

**Architecture:** Existing training pipeline (`training/train.py`) already supports all 4 encodings and both datasets. We need: (1) a batch runner script for multi-seed experiments, (2) an analysis/plotting script, (3) the actual experiment runs.

**Tech Stack:** PyTorch, matplotlib, existing quantum_encodings.py, existing train.py

---

### Task 1: Create Multi-Seed Batch Runner Script

**Files:**
- Create: `experiments/run_multiseed.py`

**Step 1: Write the batch runner**

```python
"""
Multi-seed batch runner for encoding experiments.
Runs 4 encodings x N seeds on a given dataset.

Usage:
    PYTHONIOENCODING=utf-8 python experiments/run_multiseed.py --dataset imdb --seeds 42 123 456 789 2024
    PYTHONIOENCODING=utf-8 python experiments/run_multiseed.py --dataset sst2 --seeds 42
"""
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
PROJECT_ROOT = Path(__file__).parent.parent

ENCODINGS = ["enc_angle", "enc_dense", "enc_iqp", "enc_reupload"]
BASELINES = ["baseline", "attention"]


def run_experiment(model, dataset, seed, reps=1, lr_quantum=0.05):
    """Run a single training experiment via subprocess."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "training" / "train.py"),
        "--model", model,
        "--dataset", dataset,
        "--seed", str(seed),
        "--reps", str(reps),
        "--lr_quantum", str(lr_quantum),
        "--subset",
        "--epochs", "30",
    ]
    print(f"\n{'='*60}")
    print(f"Running: {model} | {dataset} | seed={seed}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(cmd, env=env, capture_output=False)

    if result.returncode != 0:
        print(f"WARNING: {model} seed={seed} failed with code {result.returncode}")
        return False

    # Rename output files to include seed
    log_src = PROJECT_ROOT / "results" / "logs" / f"{model}_{dataset}_history.json"
    log_dst = PROJECT_ROOT / "results" / "logs" / f"{model}_{dataset}_seed{seed}_history.json"
    ckpt_src = PROJECT_ROOT / "checkpoints" / f"{model}_{dataset}_best.pt"
    ckpt_dst = PROJECT_ROOT / "checkpoints" / f"{model}_{dataset}_seed{seed}_best.pt"

    if log_src.exists():
        # Read, add seed info, save with new name
        with open(log_src) as f:
            data = json.load(f)
        data["config"]["seed"] = seed
        with open(log_dst, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Log saved: {log_dst.name}")

    if ckpt_src.exists():
        import shutil
        shutil.copy2(ckpt_src, ckpt_dst)
        print(f"  Checkpoint saved: {ckpt_dst.name}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="imdb", choices=["imdb", "sst2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2024])
    parser.add_argument("--include_baselines", action="store_true",
                        help="Also run baseline and attention models")
    args = parser.parse_args()

    models = list(ENCODINGS)
    if args.include_baselines:
        models = BASELINES + models

    total = len(models) * len(args.seeds)
    done = 0

    print(f"Total experiments: {total} ({len(models)} models x {len(args.seeds)} seeds)")
    print(f"Models: {models}")
    print(f"Seeds: {args.seeds}")
    print(f"Dataset: {args.dataset}")

    results_summary = []

    for seed in args.seeds:
        for model in models:
            done += 1
            print(f"\n[{done}/{total}] ", end="")
            success = run_experiment(model, args.dataset, seed)
            results_summary.append({
                "model": model, "seed": seed, "dataset": args.dataset,
                "success": success
            })

    # Save summary
    summary_path = PROJECT_ROOT / "results" / "logs" / f"multiseed_{args.dataset}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    print(f"Done: {sum(1 for r in results_summary if r['success'])}/{total} succeeded")


if __name__ == "__main__":
    main()
```

**Step 2: Verify script syntax**

Run: `cd "C:\Users\gokss\OneDrive\Masaüstü\Kuantum\quantum-attention" && PYTHONIOENCODING=utf-8 python -c "import ast; ast.parse(open('experiments/run_multiseed.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

---

### Task 2: Modify train.py to Support Seed-Specific Output Filenames

**Files:**
- Modify: `training/train.py`

**Problem:** Current train.py always saves to `{model}_{dataset}_best.pt` and `{model}_{dataset}_history.json`. For multi-seed runs, we need seed in the filename so results don't overwrite each other.

**Step 1: Add --output_suffix argument to argparse in train.py**

In `main()` function, add after line 452 (`parser.add_argument("--subset"...)`):
```python
parser.add_argument("--suffix", type=str, default="", help="Output filename suffix (e.g., _seed123)")
```

**Step 2: Pass suffix through config**

Add to `TrainingConfig` dataclass in `training/config.py`:
```python
output_suffix: str = ""  # e.g., "_seed123" for multi-seed experiments
```

**Step 3: Update checkpoint/log filenames in train() to use suffix**

In `train()` function, change the checkpoint save line (around line 379):
```python
ckpt_path = ckpt_dir / f"{config.model_type}_{config.dataset}{config.output_suffix}_best.pt"
```

And the log save line (around line 419):
```python
log_path = log_dir / f"{config.model_type}_{config.dataset}{config.output_suffix}_history.json"
```

And the checkpoint load line (around line 400):
```python
ckpt_path = ckpt_dir / f"{config.model_type}_{config.dataset}{config.output_suffix}_best.pt"
```

**Step 4: Wire suffix in main()**

```python
config = TrainingConfig(
    ...
    output_suffix=args.suffix,
)
```

---

### Task 3: Run Multi-Seed IMDb Experiments (Phase 7A)

**Files:** No new files. Uses `experiments/run_multiseed.py`

**Step 1: Run seeds 42, 123, 456, 789, 2024 for all 4 encodings on IMDb**

Run (sequential — max 1 at a time to avoid page file issues):
```bash
cd "C:\Users\gokss\OneDrive\Masaüstü\Kuantum\quantum-attention"
PYTHONIOENCODING=utf-8 python experiments/run_multiseed.py --dataset imdb --seeds 42 123 456 789 2024
```

Expected output: 20 experiments (4 encodings x 5 seeds), each ~8-13 minutes on CPU.
Total time estimate: ~3-4 hours.

**NOTE:** seed=42 results already exist from Phase 6. The script will overwrite them (same config, same result expected).

**Step 2: Verify all 20 log files exist**

```bash
ls results/logs/enc_*_imdb_seed*_history.json | wc -l
```
Expected: 20

---

### Task 4: Run SST-2 Experiments (Phase 7B)

**Step 1: Run 4 encodings + 2 baselines on SST-2 with seed=42**

```bash
PYTHONIOENCODING=utf-8 python experiments/run_multiseed.py --dataset sst2 --seeds 42 --include_baselines
```

Expected: 6 experiments (4 enc + baseline + attention), each ~5-13 minutes.

**Step 2: Verify log files**

```bash
ls results/logs/*_sst2_seed42_history.json
```
Expected: 6 files

---

### Task 5: Create Analysis and Plotting Script

**Files:**
- Create: `experiments/paper_analysis.py`

This script:
1. Collects all multi-seed results from JSON logs
2. Computes mean/std accuracy per encoding
3. Loads barren plateau data for gradient variance
4. Generates 4 paper-quality figures
5. Outputs a summary table

```python
"""
Paper Analysis & Figure Generation
===================================
Collects multi-seed results, barren plateau data, and generates paper figures.

Usage:
    PYTHONIOENCODING=utf-8 python experiments/paper_analysis.py
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
PROJECT_ROOT = Path(__file__).parent.parent

ENCODING_NAMES = {
    "enc_angle": "Angle",
    "enc_dense": "Dense Angle",
    "enc_iqp": "IQP",
    "enc_reupload": "Re-uploading",
}
ENCODING_COLORS = {
    "enc_angle": "#2196F3",
    "enc_dense": "#4CAF50",
    "enc_iqp": "#FF9800",
    "enc_reupload": "#F44336",
}
SEEDS = [42, 123, 456, 789, 2024]


def load_multiseed_results(dataset="imdb"):
    """Load all multi-seed experiment results."""
    log_dir = PROJECT_ROOT / "results" / "logs"
    results = {}  # {encoding: [test_acc_seed1, test_acc_seed2, ...]}

    for enc in ENCODING_NAMES:
        accs = []
        for seed in SEEDS:
            log_path = log_dir / f"{enc}_{dataset}_seed{seed}_history.json"
            if not log_path.exists():
                # Try without seed suffix (old format)
                log_path = log_dir / f"{enc}_{dataset}_history.json"
            if log_path.exists():
                with open(log_path) as f:
                    data = json.load(f)
                accs.append(data["test_acc"])
        if accs:
            results[enc] = accs
    return results


def load_barren_plateau_data():
    """Load gradient variance data."""
    bp_path = PROJECT_ROOT / "results" / "logs" / "barren_plateau_analysis.json"
    with open(bp_path) as f:
        return json.load(f)


def figure1_variance_vs_accuracy(imdb_results, bp_data, save_path):
    """
    Figure 1: Gradient Variance vs Test Accuracy scatter plot.
    Main finding visualization.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Extract gradient variance from barren plateau data
    variance_map = {}
    for item in bp_data["encoding_comparison"]:
        enc_key = {
            "AngleEncodingCircuit": "enc_angle",
            "DenseAngleEncodingCircuit": "enc_dense",
            "IQPEncodingCircuit": "enc_iqp",
            "DataReuploadingCircuit": "enc_reupload",
        }.get(item["encoding"])
        if enc_key:
            variance_map[enc_key] = item["mean_variance"]

    for enc in ENCODING_NAMES:
        if enc in imdb_results and enc in variance_map:
            mean_acc = np.mean(imdb_results[enc]) * 100
            std_acc = np.std(imdb_results[enc]) * 100
            var = variance_map[enc]

            ax.errorbar(var, mean_acc, yerr=std_acc, fmt='o', color=ENCODING_COLORS[enc],
                       markersize=12, capsize=5, capthick=2, linewidth=2,
                       label=f"{ENCODING_NAMES[enc]}")

    ax.set_xlabel("Gradient Variance (log scale)", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("Gradient Trainability vs NLP Accuracy", fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {save_path}")


def figure2_training_curves(dataset="imdb", seed=42, save_path=None):
    """
    Figure 2: Training curves (loss and accuracy) for all encodings.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    log_dir = PROJECT_ROOT / "results" / "logs"

    for enc in ENCODING_NAMES:
        log_path = log_dir / f"{enc}_{dataset}_seed{seed}_history.json"
        if not log_path.exists():
            log_path = log_dir / f"{enc}_{dataset}_history.json"
        if not log_path.exists():
            continue

        with open(log_path) as f:
            data = json.load(f)

        epochs = range(1, len(data["train_loss"]) + 1)
        color = ENCODING_COLORS[enc]
        label = ENCODING_NAMES[enc]

        axes[0].plot(epochs, data["train_loss"], color=color, label=label, linewidth=2)
        axes[1].plot(epochs, [a * 100 for a in data["val_acc"]],
                    color=color, label=label, linewidth=2)

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Training Loss", fontsize=12)
    axes[0].set_title("Training Loss Curves", fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Validation Accuracy (%)", fontsize=12)
    axes[1].set_title("Validation Accuracy Curves", fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {save_path}")


def figure3_scaling_analysis(bp_data, save_path):
    """
    Figure 3: Qubit scaling and depth scaling gradient variance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Qubit scaling
    qubits = [item["n_qubits"] for item in bp_data["qubit_scaling"]]
    variances = [item["mean_variance"] for item in bp_data["qubit_scaling"]]
    axes[0].plot(qubits, variances, 'o-', color='#2196F3', markersize=10, linewidth=2)
    axes[0].set_xlabel("Number of Qubits", fontsize=12)
    axes[0].set_ylabel("Gradient Variance", fontsize=12)
    axes[0].set_title("Qubit Scaling (Angle Encoding, reps=1)", fontsize=13, fontweight='bold')
    axes[0].set_xticks(qubits)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    # Depth scaling
    reps_list = [item["reps"] for item in bp_data["depth_scaling"]]
    variances = [item["mean_variance"] for item in bp_data["depth_scaling"]]
    axes[1].plot(reps_list, variances, 's-', color='#F44336', markersize=10, linewidth=2)
    axes[1].set_xlabel("Circuit Depth (reps)", fontsize=12)
    axes[1].set_ylabel("Gradient Variance", fontsize=12)
    axes[1].set_title("Depth Scaling (Angle Encoding, 8 qubits)", fontsize=13, fontweight='bold')
    axes[1].set_xticks(reps_list)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {save_path}")


def figure4_multiseed_barplot(imdb_results, sst2_results, save_path):
    """
    Figure 4: Multi-seed accuracy bar plot with error bars, both datasets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    encodings = list(ENCODING_NAMES.keys())
    x = np.arange(len(encodings))
    width = 0.6

    for ax, results, title in [(axes[0], imdb_results, "IMDb"), (axes[1], sst2_results, "SST-2")]:
        means = []
        stds = []
        colors = []
        labels = []
        for enc in encodings:
            if enc in results:
                means.append(np.mean(results[enc]) * 100)
                stds.append(np.std(results[enc]) * 100)
            else:
                means.append(0)
                stds.append(0)
            colors.append(ENCODING_COLORS[enc])
            labels.append(ENCODING_NAMES[enc])

        bars = ax.bar(x, means, width, yerr=stds, color=colors,
                     capsize=5, edgecolor='black', linewidth=0.5, alpha=0.85)

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.5,
                       f'{mean:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel("Encoding Strategy", fontsize=12)
        ax.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax.set_title(f"{title} Sentiment Classification", fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(50, 80)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {save_path}")


def print_summary_table(imdb_results, sst2_results, bp_data):
    """Print formatted summary table for the paper."""
    print("\n" + "=" * 80)
    print("PAPER SUMMARY TABLE")
    print("=" * 80)

    # Extract gradient variance
    variance_map = {}
    for item in bp_data["encoding_comparison"]:
        enc_key = {
            "AngleEncodingCircuit": "enc_angle",
            "DenseAngleEncodingCircuit": "enc_dense",
            "IQPEncodingCircuit": "enc_iqp",
            "DataReuploadingCircuit": "enc_reupload",
        }.get(item["encoding"])
        if enc_key:
            variance_map[enc_key] = item["mean_variance"]

    print(f"\n{'Encoding':<15} | {'Grad Var':>8} | {'IMDb Acc':>12} | {'SST-2 Acc':>12} | {'Qubits':>6} | {'Params':>6}")
    print("-" * 75)

    for enc in ENCODING_NAMES:
        var_str = f"{variance_map.get(enc, 0):.3f}"

        if enc in imdb_results:
            imdb_mean = np.mean(imdb_results[enc]) * 100
            imdb_std = np.std(imdb_results[enc]) * 100
            imdb_str = f"{imdb_mean:.1f} +/- {imdb_std:.1f}%"
        else:
            imdb_str = "N/A"

        if enc in sst2_results:
            sst2_mean = np.mean(sst2_results[enc]) * 100
            sst2_str = f"{sst2_mean:.1f}%"
        else:
            sst2_str = "N/A"

        qubits = "4" if enc == "enc_dense" else "8"
        params = "26" if enc == "enc_dense" else "50"

        print(f"{ENCODING_NAMES[enc]:<15} | {var_str:>8} | {imdb_str:>12} | {sst2_str:>12} | {qubits:>6} | {params:>6}")


def main():
    print("Paper Analysis & Figure Generation")
    print("=" * 50)

    plot_dir = PROJECT_ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    imdb_results = load_multiseed_results("imdb")
    sst2_results = load_multiseed_results("sst2")
    bp_data = load_barren_plateau_data()

    print(f"  IMDb: {sum(len(v) for v in imdb_results.values())} results across {len(imdb_results)} encodings")
    print(f"  SST-2: {sum(len(v) for v in sst2_results.values())} results across {len(sst2_results)} encodings")

    # Generate figures
    print("\nGenerating figures...")
    figure1_variance_vs_accuracy(imdb_results, bp_data,
                                  plot_dir / "fig1_variance_vs_accuracy.png")
    figure2_training_curves("imdb", 42,
                           plot_dir / "fig2_training_curves.png")
    figure3_scaling_analysis(bp_data,
                            plot_dir / "fig3_scaling_analysis.png")
    figure4_multiseed_barplot(imdb_results, sst2_results,
                             plot_dir / "fig4_multiseed_accuracy.png")

    # Summary table
    print_summary_table(imdb_results, sst2_results, bp_data)

    # Save numeric summary as JSON
    summary = {
        "imdb": {enc: {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "n": len(accs)}
                 for enc, accs in imdb_results.items()},
        "sst2": {enc: {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "n": len(accs)}
                 for enc, accs in sst2_results.items()},
    }
    summary_path = plot_dir / "paper_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nNumeric summary: {summary_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Verify syntax**

Run: `PYTHONIOENCODING=utf-8 python -c "import ast; ast.parse(open('experiments/paper_analysis.py').read()); print('OK')"`

---

### Task 6: Run Analysis and Generate Figures

**Prerequisite:** Tasks 3 and 4 must be complete (experiment data available).

**Step 1: Run the analysis script**

```bash
cd "C:\Users\gokss\OneDrive\Masaüstü\Kuantum\quantum-attention"
PYTHONIOENCODING=utf-8 python experiments/paper_analysis.py
```

**Step 2: Verify all 4 figures were generated**

```bash
ls results/plots/fig*.png
```

Expected:
- `fig1_variance_vs_accuracy.png` — Main finding
- `fig2_training_curves.png` — Training dynamics
- `fig3_scaling_analysis.png` — Qubit/depth scaling
- `fig4_multiseed_accuracy.png` — Bar chart with error bars

**Step 3: Visually inspect figures**

Read the PNG files to verify they look correct and paper-quality.

---

### Task 7: Update CLAUDE.md with Final Results

**Files:**
- Modify: `CLAUDE.md`

After experiments complete, update Section 3 with multi-seed means/stds and SST-2 results.
Update Section 11 TODO list marking completed items.

---

### Task 8: Update developments.md with Phase 7

**Files:**
- Modify: `developments.md`

Add Phase 7 section documenting:
- Multi-seed validation results (means, stds)
- SST-2 generalization results
- Generated figures
- Final paper-ready data summary
