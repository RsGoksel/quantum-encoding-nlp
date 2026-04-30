"""
Paper Analysis & Figure Generation
===================================
Multi-seed sonuclari + barren plateau verisi + IBM QPU sonuclari ile
bildiri icin paper-quality figurler uretir.

Usage:
    PYTHONIOENCODING=utf-8 python experiments/paper_analysis.py
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    results = {}

    for enc in ENCODING_NAMES:
        accs = []
        for seed in SEEDS:
            # Try seed-specific first
            log_path = log_dir / f"{enc}_{dataset}_seed{seed}_history.json"
            if not log_path.exists():
                log_path = log_dir / f"{enc}_{dataset}_history.json"
            if log_path.exists():
                with open(log_path) as f:
                    data = json.load(f)
                if "test_acc" in data:
                    accs.append(data["test_acc"])
        if accs:
            results[enc] = accs
    return results


def load_barren_plateau_data():
    """Load gradient variance data."""
    bp_path = PROJECT_ROOT / "results" / "logs" / "barren_plateau_analysis.json"
    if not bp_path.exists():
        print("  WARNING: barren_plateau_analysis.json not found")
        return None
    with open(bp_path) as f:
        return json.load(f)


def load_ibm_results():
    """Load IBM QPU encoding comparison results."""
    ibm_path = PROJECT_ROOT / "results" / "logs" / "ibm_encoding_comparison.json"
    if not ibm_path.exists():
        return None
    with open(ibm_path) as f:
        return json.load(f)


def get_variance_map(bp_data):
    """Extract gradient variance per encoding from barren plateau data."""
    if bp_data is None:
        return {}
    # Map barren plateau JSON keys to encoding keys
    key_map = {
        "enc_angle_8f_r1": "enc_angle",
        "enc_dense_angle_8f_r1": "enc_dense",
        "enc_iqp_8f_r1": "enc_iqp",
        "enc_re-uploading_8f_r1": "enc_reupload",
    }
    variance_map = {}
    for bp_key, enc_key in key_map.items():
        if bp_key in bp_data:
            variance_map[enc_key] = bp_data[bp_key]["mean_grad_var"]
    return variance_map


# ===========================================================================
# Figure 1: Gradient Variance vs Accuracy (MAIN FINDING)
# ===========================================================================

def figure1_variance_vs_accuracy(imdb_results, bp_data, save_path):
    """Main finding: gradient variance predicts accuracy."""
    variance_map = get_variance_map(bp_data)
    if not variance_map:
        print("  Skipping Figure 1 (no barren plateau data)")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))

    for enc in ENCODING_NAMES:
        if enc in imdb_results and enc in variance_map:
            mean_acc = np.mean(imdb_results[enc]) * 100
            std_acc = np.std(imdb_results[enc]) * 100 if len(imdb_results[enc]) > 1 else 0
            var = variance_map[enc]

            ax.errorbar(var, mean_acc, yerr=std_acc, fmt='o',
                       color=ENCODING_COLORS[enc], markersize=14,
                       capsize=6, capthick=2, linewidth=2, markeredgecolor='black',
                       markeredgewidth=1, label=ENCODING_NAMES[enc])

    ax.set_xlabel("Gradient Variance (log scale)", fontsize=14)
    ax.set_ylabel("Test Accuracy (%)", fontsize=14)
    ax.set_title("Gradient Trainability vs NLP Task Accuracy", fontsize=15, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure 1 saved: {save_path.name}")


# ===========================================================================
# Figure 2: Training Curves
# ===========================================================================

def figure2_training_curves(dataset="imdb", seed=42, save_path=None):
    """Training dynamics comparison across encodings."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
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

        axes[0].plot(epochs, data["train_loss"], color=color, label=label, linewidth=2.5)
        axes[1].plot(epochs, [a * 100 for a in data["val_acc"]],
                    color=color, label=label, linewidth=2.5)

    for ax, ylabel, title in [(axes[0], "Training Loss", "Training Loss"),
                               (axes[1], "Validation Accuracy (%)", "Validation Accuracy")]:
        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure 2 saved: {save_path.name}")


# ===========================================================================
# Figure 3: Scaling Analysis
# ===========================================================================

def figure3_scaling_analysis(bp_data, save_path):
    """Qubit and depth scaling of gradient variance."""
    if bp_data is None:
        print("  Skipping Figure 3 (no barren plateau data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Qubit scaling: angle_4q_r1, angle_6q_r1, angle_8q_r1, angle_10q_r1
    qubit_keys = [("angle_4q_r1", 4), ("angle_6q_r1", 6), ("angle_8q_r1", 8), ("angle_10q_r1", 10)]
    qubits = [q for k, q in qubit_keys if k in bp_data]
    variances = [bp_data[k]["mean_grad_var"] for k, q in qubit_keys if k in bp_data]

    if qubits:
        axes[0].plot(qubits, variances, 'o-', color='#2196F3', markersize=12, linewidth=2.5,
                    markeredgecolor='black', markeredgewidth=1)
        axes[0].set_xlabel("Number of Qubits", fontsize=13)
        axes[0].set_ylabel("Gradient Variance", fontsize=13)
        axes[0].set_title("Qubit Scaling (Angle, reps=1)", fontsize=14, fontweight='bold')
        axes[0].set_xticks(qubits)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(bottom=0, top=max(variances) * 1.2)

    # Depth scaling: angle_8q_r1, angle_8q_r2, angle_8q_r3
    depth_keys = [("angle_8q_r1", 1), ("angle_8q_r2", 2), ("angle_8q_r3", 3)]
    reps_list = [r for k, r in depth_keys if k in bp_data]
    variances = [bp_data[k]["mean_grad_var"] for k, r in depth_keys if k in bp_data]

    if reps_list:
        axes[1].plot(reps_list, variances, 's-', color='#F44336', markersize=12, linewidth=2.5,
                    markeredgecolor='black', markeredgewidth=1)
        axes[1].set_xlabel("Circuit Depth (reps)", fontsize=13)
        axes[1].set_ylabel("Gradient Variance", fontsize=13)
        axes[1].set_title("Depth Scaling (Angle, 8 qubits)", fontsize=14, fontweight='bold')
        axes[1].set_xticks(reps_list)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(bottom=0, top=max(variances) * 1.2)

    for ax in axes:
        ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure 3 saved: {save_path.name}")


# ===========================================================================
# Figure 4: Multi-seed Accuracy Bar Chart
# ===========================================================================

def figure4_multiseed_barplot(imdb_results, sst2_results, save_path):
    """Bar chart with error bars for both datasets."""
    encodings = list(ENCODING_NAMES.keys())
    n_enc = len(encodings)

    has_sst2 = any(enc in sst2_results for enc in encodings)
    n_plots = 2 if has_sst2 else 1

    fig, axes = plt.subplots(1, n_plots, figsize=(6.5 * n_plots, 5.5))
    if n_plots == 1:
        axes = [axes]

    datasets = [("IMDb", imdb_results)]
    if has_sst2:
        datasets.append(("SST-2", sst2_results))

    for ax, (title, results) in zip(axes, datasets):
        x = np.arange(n_enc)
        means, stds, colors, labels = [], [], [], []

        for enc in encodings:
            if enc in results:
                means.append(np.mean(results[enc]) * 100)
                stds.append(np.std(results[enc]) * 100 if len(results[enc]) > 1 else 0)
            else:
                means.append(0)
                stds.append(0)
            colors.append(ENCODING_COLORS[enc])
            labels.append(ENCODING_NAMES[enc])

        bars = ax.bar(x, means, 0.6, yerr=stds, color=colors,
                     capsize=6, edgecolor='black', linewidth=0.8, alpha=0.85)

        for bar, m, s in zip(bars, means, stds):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + s + 0.5,
                       f'{m:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xlabel("Encoding Strategy", fontsize=13)
        ax.set_ylabel("Test Accuracy (%)", fontsize=13)
        ax.set_title(f"{title} Sentiment Classification", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(50, 80)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure 4 saved: {save_path.name}")


# ===========================================================================
# Figure 5: IBM QPU vs Simulator (if available)
# ===========================================================================

def figure5_ibm_comparison(ibm_data, save_path):
    """Simulator vs QPU accuracy per encoding."""
    if ibm_data is None:
        print("  Skipping Figure 5 (no IBM QPU data)")
        return

    encodings_data = ibm_data.get("encodings", {})
    if not encodings_data:
        print("  Skipping Figure 5 (empty IBM data)")
        return

    enc_names = []
    sim_accs = []
    qpu_accs = []
    colors = []

    for short_name, full_name in [("angle", "Angle"), ("dense", "Dense Angle"),
                                   ("iqp", "IQP"), ("reupload", "Re-uploading")]:
        if short_name in encodings_data:
            enc_names.append(full_name)
            sim_accs.append(encodings_data[short_name]["sim_accuracy"] * 100)
            qpu_accs.append(encodings_data[short_name]["qpu_accuracy"] * 100)
            colors.append(ENCODING_COLORS.get(f"enc_{short_name}", "#999"))

    if not enc_names:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    x = np.arange(len(enc_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, sim_accs, width, label='Simulator',
                   color=[c + '88' for c in colors], edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, qpu_accs, width, label=f'IBM QPU ({ibm_data.get("backend", "?")})',
                   color=colors, edgecolor='black', linewidth=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                   f'{h:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel("Encoding Strategy", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Simulator vs IBM QPU: Encoding Noise Resilience", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(enc_names, fontsize=11)
    ax.legend(fontsize=12)
    ax.set_ylim(30, 90)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure 5 saved: {save_path.name}")


# ===========================================================================
# Summary table
# ===========================================================================

def print_summary(imdb_results, sst2_results, bp_data, ibm_data):
    """Print formatted summary for the paper."""
    variance_map = get_variance_map(bp_data)

    print("\n" + "=" * 90)
    print("PAPER SUMMARY TABLE — Bildiri icin ana tablo")
    print("=" * 90)

    header = f"{'Encoding':<15} | {'Grad Var':>8} | {'IMDb Acc':>16} | {'SST-2 Acc':>16} | {'QPU Acc':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for enc in ENCODING_NAMES:
        parts = [f"{ENCODING_NAMES[enc]:<15}"]

        # Gradient variance
        var = variance_map.get(enc, None)
        parts.append(f"{var:>8.3f}" if var else f"{'N/A':>8}")

        # IMDb
        if enc in imdb_results:
            m = np.mean(imdb_results[enc]) * 100
            s = np.std(imdb_results[enc]) * 100 if len(imdb_results[enc]) > 1 else 0
            parts.append(f"{m:>5.1f} +/- {s:>4.1f}%" if s > 0 else f"{m:>11.1f}%    ")
        else:
            parts.append(f"{'N/A':>16}")

        # SST-2
        if enc in sst2_results:
            m = np.mean(sst2_results[enc]) * 100
            parts.append(f"{m:>11.1f}%    ")
        else:
            parts.append(f"{'N/A':>16}")

        # QPU
        short = enc.replace("enc_", "")
        if ibm_data and short in ibm_data.get("encodings", {}):
            qpu_acc = ibm_data["encodings"][short]["qpu_accuracy"] * 100
            parts.append(f"{qpu_acc:>7.1f}%")
        else:
            parts.append(f"{'pending':>8}")

        print(" | ".join(parts))

    # Save numeric summary
    summary = {}
    for dataset_name, results in [("imdb", imdb_results), ("sst2", sst2_results)]:
        summary[dataset_name] = {}
        for enc, accs in results.items():
            summary[dataset_name][enc] = {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)) if len(accs) > 1 else 0,
                "n_seeds": len(accs),
                "accs": [float(a) for a in accs],
            }

    summary_path = PROJECT_ROOT / "results" / "plots" / "paper_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nNumeric summary saved: {summary_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("Paper Analysis & Figure Generation")
    print("=" * 60)

    plot_dir = PROJECT_ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    imdb_results = load_multiseed_results("imdb")
    sst2_results = load_multiseed_results("sst2")
    bp_data = load_barren_plateau_data()
    ibm_data = load_ibm_results()

    for name, res in [("IMDb", imdb_results), ("SST-2", sst2_results)]:
        total = sum(len(v) for v in res.values())
        print(f"  {name}: {total} results across {len(res)} encodings")
    print(f"  Barren plateau: {'loaded' if bp_data else 'NOT FOUND'}")
    print(f"  IBM QPU: {'loaded' if ibm_data else 'NOT YET (run ibm_encoding_inference.py first)'}")

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
    figure5_ibm_comparison(ibm_data,
                           plot_dir / "fig5_ibm_comparison.png")

    # Summary
    print_summary(imdb_results, sst2_results, bp_data, ibm_data)


if __name__ == "__main__":
    main()
