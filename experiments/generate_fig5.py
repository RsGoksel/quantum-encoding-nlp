"""
Figure 5: IBM QPU vs Simulator Accuracy per Encoding
=====================================================
Grouped bar chart comparing simulator and QPU accuracy for all 4 encodings.
Transpiled depth annotated above each group.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Load IBM QPU results
with open(PROJECT_ROOT / "results" / "logs" / "ibm_encoding_comparison.json") as f:
    data = json.load(f)

encodings = ["angle", "dense", "iqp", "reupload"]
labels = ["Angle", "Dense Angle", "IQP", "Re-uploading"]
depths = [19, 18, 101, 21]

sim_acc = [data["encodings"][e]["sim_accuracy"] * 100 for e in encodings]
qpu_acc = [data["encodings"][e]["qpu_accuracy"] * 100 for e in encodings]
noise_gaps = [sim_acc[i] - qpu_acc[i] for i in range(4)]

x = np.arange(len(encodings))
width = 0.35

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

bars_sim = ax.bar(x - width/2, sim_acc, width, label="Simulator",
                  color="#4C72B0", edgecolor="black", linewidth=0.8)
bars_qpu = ax.bar(x + width/2, qpu_acc, width, label="IBM QPU (ibm_fez)",
                  color="#DD8452", edgecolor="black", linewidth=0.8)

# Add value labels on bars
for bar in bars_sim:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bars_qpu:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add transpiled depth annotations
for i, (depth, gap) in enumerate(zip(depths, noise_gaps)):
    max_h = max(sim_acc[i], qpu_acc[i])
    gap_text = f"Gap: {gap:+.0f}%" if abs(gap) > 0.01 else "Gap: 0%"
    color = "green" if abs(gap) < 0.01 else ("orange" if abs(gap) <= 10 else "red")
    ax.text(x[i], max_h + 8, f"Depth: {depth}\n{gap_text}",
            ha='center', va='bottom', fontsize=9, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor=color, alpha=0.8))

ax.set_xlabel("Encoding Strategy", fontsize=13)
ax.set_ylabel("Test Accuracy (%)", fontsize=13)
ax.set_title("Simulator vs. IBM QPU Accuracy\n(ibm_fez, 156 qubits, Eagle r3, 10 samples)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylim(0, 100)
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Add horizontal line at 50% (random)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random chance')
ax.text(3.5, 51, 'Random chance', fontsize=9, color='gray', ha='right')

plt.tight_layout()

output_path = PROJECT_ROOT / "results" / "plots" / "fig5_ibm_qpu_comparison.png"
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Figure 5 saved: {output_path}")
plt.close()

# Also generate a second version: noise gap bar chart
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))

colors = ['#2ecc71' if abs(g) < 0.01 else '#e67e22' if abs(g) <= 10 else '#e74c3c' for g in noise_gaps]
bars = ax2.bar(x, noise_gaps, 0.6, color=colors, edgecolor='black', linewidth=0.8)

for bar, gap in zip(bars, noise_gaps):
    height = bar.get_height()
    va = 'bottom' if height >= 0 else 'top'
    offset = 0.5 if height >= 0 else -0.5
    ax2.text(bar.get_x() + bar.get_width()/2., height + offset,
             f'{gap:+.0f}%', ha='center', va=va, fontsize=12, fontweight='bold')

ax2.set_xlabel("Encoding Strategy", fontsize=13)
ax2.set_ylabel("Noise Gap (Sim - QPU) %", fontsize=13)
ax2.set_title("Noise-Induced Accuracy Gap per Encoding\n(Positive = QPU worse, Negative = QPU better)", fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=12)
ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_ylim(-15, 15)
ax2.grid(axis='y', alpha=0.3)

# Add depth as secondary info
for i, depth in enumerate(depths):
    ax2.text(x[i], -13, f"Depth: {depth}", ha='center', fontsize=9, color='gray')

plt.tight_layout()
output_path2 = PROJECT_ROOT / "results" / "plots" / "fig5b_noise_gap.png"
plt.savefig(output_path2, dpi=200, bbox_inches='tight')
print(f"Figure 5b saved: {output_path2}")
plt.close()
