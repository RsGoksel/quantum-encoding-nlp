"""
Barren Plateau Gradient Variance Analysis
==========================================
Farkli encoding stratejileri ve qubit sayilari icin
gradient variance olcumu.

Barren plateau nedir?
- Kuantum circuit derinligi arttikca, parametre gradyanlari
  ustsel olarak kucule bilir (variance → 0)
- Bu, training'i imkansiz kilar (gradient yok → ogrenme yok)
- McClean et al. (2018): Var[dC/dtheta] ~ O(2^-n) random circuit'ler icin

Bu script:
1. Her encoding icin 100 random initialization yapar
2. Her initialization icin gradient hesaplar
3. Gradient variance'i raporlar
4. Encoding'ler arasi karsilastirma

Kullanim:
    PYTHONIOENCODING=utf-8 python experiments/barren_plateau.py
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.quantum_encodings import (
    AngleEncodingCircuit, DenseAngleEncodingCircuit,
    IQPEncodingCircuit, DataReuploadingCircuit,
)


def compute_gradient_variance(circuit_class, n_qubits, reps, n_samples=100,
                              n_inputs=10, seed=42, n_features=None):
    """
    Gradient variance hesapla.

    n_samples kez random init yap, her init icin n_inputs input uzerinden
    ortalama gradient hesapla, sonra bu gradient'lerin variance'ini raporla.
    """
    torch.manual_seed(seed)
    input_dim = n_features if n_features else n_qubits

    all_grad_norms = []

    for s in range(n_samples):
        # Random initialization
        if n_features:
            circuit = circuit_class(n_features=n_features, reps=reps, identity_init=False)
        else:
            circuit = circuit_class(n_qubits=n_qubits, reps=reps, identity_init=False)
        # Manually re-randomize
        with torch.no_grad():
            circuit.thetas.copy_(torch.rand_like(circuit.thetas) * 2 * np.pi)

        # Random inputs
        x = torch.randn(n_inputs, input_dim)

        # Forward + backward
        circuit.zero_grad()
        out = circuit(x)
        loss = out.sum()
        loss.backward()

        grad = circuit.thetas.grad
        if grad is not None:
            all_grad_norms.append(grad.clone().numpy())

    # Stack: (n_samples, n_params)
    grads = np.stack(all_grad_norms)

    # Per-parameter variance
    per_param_var = np.var(grads, axis=0)  # (n_params,)

    return {
        "mean_grad_var": float(np.mean(per_param_var)),
        "max_grad_var": float(np.max(per_param_var)),
        "min_grad_var": float(np.min(per_param_var)),
        "mean_grad_abs": float(np.mean(np.abs(grads))),
        "per_param_var": per_param_var.tolist(),
        "n_params": len(per_param_var),
    }


def main():
    print("=" * 60)
    print("Barren Plateau Gradient Variance Analysis")
    print("=" * 60)

    results = {}

    # Encoding'ler
    encodings = {
        "angle": (AngleEncodingCircuit, {}),
        "dense_angle": (DenseAngleEncodingCircuit, {"n_features": 8}),
        "iqp": (IQPEncodingCircuit, {}),
        "re-uploading": (DataReuploadingCircuit, {}),
    }

    # 1. Encoding karsilastirmasi (8 feature input, reps=1)
    print("\n--- Encoding Karsilastirmasi (8 feature, reps=1) ---")
    for enc_name, (cls, kwargs) in encodings.items():
        if enc_name == "dense_angle":
            res = compute_gradient_variance(cls, n_qubits=4, reps=1, n_features=8)
        else:
            res = compute_gradient_variance(cls, n_qubits=8, reps=1)
        results[f"enc_{enc_name}_8f_r1"] = res
        n_q = 4 if enc_name == "dense_angle" else 8
        print(f"  {enc_name:15s}: {n_q}q, mean_var={res['mean_grad_var']:.6f}, "
              f"mean|grad|={res['mean_grad_abs']:.6f}, params={res['n_params']}")

    # 2. Qubit scaling (angle encoding, reps=1)
    print("\n--- Qubit Scaling (angle encoding, reps=1) ---")
    for n_q in [4, 6, 8, 10]:
        res = compute_gradient_variance(AngleEncodingCircuit, n_qubits=n_q, reps=1)
        results[f"angle_{n_q}q_r1"] = res
        print(f"  {n_q} qubits: mean_var={res['mean_grad_var']:.6f}, "
              f"mean|grad|={res['mean_grad_abs']:.6f}, params={res['n_params']}")

    # 3. Depth scaling (angle encoding, 8 qubits)
    print("\n--- Depth Scaling (angle encoding, 8 qubits) ---")
    for r in [1, 2, 3]:
        res = compute_gradient_variance(AngleEncodingCircuit, n_qubits=8, reps=r)
        results[f"angle_8q_r{r}"] = res
        print(f"  reps={r}: mean_var={res['mean_grad_var']:.6f}, "
              f"mean|grad|={res['mean_grad_abs']:.6f}, params={res['n_params']}")

    # Kaydet
    out_path = PROJECT_ROOT / "results" / "logs" / "barren_plateau_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSonuclar: {out_path}")
    print("Bitti!")


if __name__ == "__main__":
    main()
