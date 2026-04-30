"""
IBM QPU Inference — All 4 Encoding Strategies
===============================================
Egitilmis 4 farkli encoding modelini IBM gercek quantum bilgisayarinda calistirir.
Simulator vs QPU accuracy farki = encoding'in noise resilience'i.

Bu script bildirinin EN ONEMLI deneyini yapar:
"Hangi encoding gercek quantum donanminda en az noise'dan etkilenir?"

Kullanim:
    PYTHONIOENCODING=utf-8 python experiments/ibm_encoding_inference.py

QPU suresi tahmini:
    5 sample x 4 encoding x ~8 observable = ~160 PUB
    ~4-5 job x ~60s = ~4-5 dakika (9 dk limitte sigar)
"""

import os
import sys
import json
import time
import math
import torch
import numpy as np
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ===========================================================================
# 1. Model ve veri yukleme
# ===========================================================================

def load_encoding_model(encoding_name, dataset="imdb"):
    """Egitilmis encoding modelini yukle."""
    from models.quantum_encodings import QuantumEncodingModel

    # Encoding → model parametreleri
    enc_map = {
        "angle": ("angle", 8),
        "dense": ("dense_angle", 8),
        "iqp": ("iqp", 8),
        "reupload": ("re-uploading", 8),
    }
    enc_type, n_qubits = enc_map[encoding_name]

    model = QuantumEncodingModel(
        encoding=enc_type, n_qubits=n_qubits, reps=1, identity_init=True
    )

    ckpt_path = PROJECT_ROOT / "checkpoints" / f"enc_{encoding_name}_{dataset}_best.pt"
    if not ckpt_path.exists():
        # Try seed-specific
        ckpt_path = PROJECT_ROOT / "checkpoints" / f"enc_{encoding_name}_{dataset}_seed42_best.pt"

    if not ckpt_path.exists():
        print(f"  WARNING: Checkpoint not found: {ckpt_path}")
        return None, None, None, None

    ckpt = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    thetas = model.circuit.thetas.detach().numpy()
    classifier_w = model.classifier.weight.detach().numpy()
    classifier_b = model.classifier.bias.detach().numpy()

    n_qubits_actual = model.circuit.n_qubits
    print(f"  {encoding_name}: loaded (val_acc={ckpt['val_acc']:.3f}, "
          f"{n_qubits_actual}q, {len(thetas)} params)")

    return model, thetas, classifier_w, classifier_b


def load_test_data(dataset="imdb", n_samples=10):
    """Test verisini yukle."""
    data = torch.load(PROJECT_ROOT / "data" / f"{dataset}_embeddings.pt", weights_only=False)
    test_X = data["test_X"][:n_samples]
    test_y = data["test_y"][:n_samples]
    return test_X, test_y


# ===========================================================================
# 2. Simulator inference (referans)
# ===========================================================================

def simulator_inference(model, test_X, test_y):
    """PyTorch simulator'de inference."""
    model.eval()
    with torch.no_grad():
        logits = model(test_X)
        preds = logits.argmax(dim=1)
    acc = (preds == test_y).float().mean().item()
    return preds.numpy(), acc


# ===========================================================================
# 3. Qiskit circuit builders (her encoding icin ayri)
# ===========================================================================

def _build_variational_layers(qc, n_qubits, reps, var_params):
    """
    Build variational layers EXACTLY matching PyTorch QuantumCircuitBase.

    Structure per rep: Ry(theta) all qubits -> Rz(theta) all qubits -> CZ linear
    Final layer (rep == reps): Ry+Rz only, NO CZ.

    CRITICAL: Uses CZ (NOT CX/CNOT) to match PyTorch implementation.
    """
    param_idx = 0
    for rep in range(reps + 1):
        for q in range(n_qubits):
            qc.ry(var_params[param_idx], q)
            param_idx += 1
        for q in range(n_qubits):
            qc.rz(var_params[param_idx], q)
            param_idx += 1
        if rep < reps:
            for q in range(n_qubits - 1):
                qc.cz(q, q + 1)
    return param_idx


def build_angle_circuit(n_qubits=8, reps=1):
    """Angle encoding: Ry(x_i) per qubit, then variational CZ layers."""
    from qiskit.circuit import QuantumCircuit, ParameterVector

    input_params = ParameterVector("x", n_qubits)
    n_var_params = n_qubits * 2 * (reps + 1)
    var_params = ParameterVector("t", n_var_params)

    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(input_params[i], i)

    _build_variational_layers(qc, n_qubits, reps, var_params)
    return qc, input_params, var_params


def build_dense_angle_circuit(n_features=8, reps=1):
    """Dense angle: Ry(x_{2i}) + Rz(x_{2i+1}) per qubit (4 qubits for 8 features)."""
    from qiskit.circuit import QuantumCircuit, ParameterVector

    n_qubits = n_features // 2
    input_params = ParameterVector("x", n_features)
    n_var_params = n_qubits * 2 * (reps + 1)
    var_params = ParameterVector("t", n_var_params)

    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.ry(input_params[2 * q], q)
        qc.rz(input_params[2 * q + 1], q)

    _build_variational_layers(qc, n_qubits, reps, var_params)
    return qc, input_params, var_params


def build_iqp_circuit(n_qubits=8, reps=1):
    """IQP: H -> Rz(x_i) -> RZZ(x_i*x_j) nearest-neighbor, then variational CZ layers."""
    from qiskit.circuit import QuantumCircuit, ParameterVector

    input_params = ParameterVector("x", n_qubits)
    interaction_params = ParameterVector("xx", n_qubits - 1)
    n_var_params = n_qubits * 2 * (reps + 1)
    var_params = ParameterVector("t", n_var_params)

    qc = QuantumCircuit(n_qubits)
    # H gates
    for q in range(n_qubits):
        qc.h(q)
    # Rz(x_i)
    for q in range(n_qubits):
        qc.rz(input_params[q], q)
    # RZZ(x_i*x_j) = CNOT-Rz-CNOT decomposition
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
        qc.rz(interaction_params[q], q + 1)
        qc.cx(q, q + 1)

    _build_variational_layers(qc, n_qubits, reps, var_params)
    return qc, input_params, var_params, interaction_params


def build_reupload_circuit(n_qubits=8, reps=1):
    """Data re-uploading: Ry(x) before each variational layer."""
    from qiskit.circuit import QuantumCircuit, ParameterVector

    input_params = ParameterVector("x", n_qubits)
    n_var_params = n_qubits * 2 * (reps + 1)
    var_params = ParameterVector("t", n_var_params)

    qc = QuantumCircuit(n_qubits)

    # Initial encoding
    for q in range(n_qubits):
        qc.ry(input_params[q], q)

    param_idx = 0
    for rep in range(reps + 1):
        for q in range(n_qubits):
            qc.ry(var_params[param_idx], q)
            param_idx += 1
        for q in range(n_qubits):
            qc.rz(var_params[param_idx], q)
            param_idx += 1
        if rep < reps:
            for q in range(n_qubits - 1):
                qc.cz(q, q + 1)
            # Re-upload data
            for q in range(n_qubits):
                qc.ry(input_params[q], q)

    return qc, input_params, var_params


# ===========================================================================
# 4. IBM QPU inference
# ===========================================================================

def run_qpu_inference(encoding_name, thetas, classifier_w, classifier_b,
                      test_X_scaled, test_y, backend, n_samples=5):
    """
    Tek encoding icin IBM QPU inference.

    Returns: (preds, accuracy, qpu_time)
    """
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2

    print(f"\n  --- {encoding_name.upper()} encoding ---")

    # Build circuit based on encoding type
    interaction_params = None
    if encoding_name == "angle":
        qc, input_params, var_params = build_angle_circuit(8, 1)
        n_qubits = 8
    elif encoding_name == "dense":
        qc, input_params, var_params = build_dense_angle_circuit(8, 1)
        n_qubits = 4
    elif encoding_name == "iqp":
        qc, input_params, var_params, interaction_params = build_iqp_circuit(8, 1)
        n_qubits = 8
    elif encoding_name == "reupload":
        qc, input_params, var_params = build_reupload_circuit(8, 1)
        n_qubits = 8

    # Observables (Pauli-Z per qubit)
    observables = []
    for i in range(n_qubits):
        pauli_str = "I" * (n_qubits - 1 - i) + "Z" + "I" * i
        observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))

    # Transpile
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled = pm.run(qc)
    transpiled_obs = [ob.apply_layout(transpiled.layout) for ob in observables]
    print(f"  Transpiled depth: {transpiled.depth()}")

    # Prepare test data
    test_data_np = test_X_scaled[:n_samples].numpy()

    # Build PUBs
    estimator = EstimatorV2(mode=backend)
    batch_size = 5  # 5 sample * n_obs = max 40 PUB/job

    all_z = np.zeros((n_samples, n_qubits))
    t0 = time.time()

    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        pubs = []

        for i in range(batch_start, batch_end):
            param_values = {}
            # Input parameters
            for j, p in enumerate(input_params):
                if j < len(test_data_np[i]):
                    param_values[p] = float(test_data_np[i, j])

            # Interaction parameters (IQP only)
            if interaction_params is not None:
                for j in range(len(interaction_params)):
                    # x_j * x_{j+1} interaction
                    param_values[interaction_params[j]] = float(
                        test_data_np[i, j] * test_data_np[i, j + 1]
                    )

            # Variational parameters (trained thetas)
            for j, p in enumerate(var_params):
                param_values[p] = float(thetas[j])

            bound = transpiled.assign_parameters(param_values)
            for obs in transpiled_obs:
                pubs.append((bound, obs))

        job = estimator.run(pubs)
        batch_n = batch_end - batch_start
        print(f"  Job {job.job_id()}: {len(pubs)} PUBs, samples {batch_start}-{batch_end-1}")

        result = job.result()
        for i in range(batch_n):
            for q in range(n_qubits):
                idx = i * n_qubits + q
                all_z[batch_start + i, q] = result[idx].data.evs

    dt = time.time() - t0

    # Classify
    logits = all_z @ classifier_w.T + classifier_b
    preds = np.argmax(logits, axis=1)
    test_y_np = test_y[:n_samples].numpy()
    acc = (preds == test_y_np).mean()

    print(f"  QPU Accuracy: {acc:.4f} ({int((preds == test_y_np).sum())}/{n_samples})")
    print(f"  QPU Time: {dt:.1f}s")

    return preds, float(acc), dt


# ===========================================================================
# 5. Main
# ===========================================================================

def main():
    print("=" * 60)
    print("IBM QPU Inference — 4 Encoding Strategies")
    print("=" * 60)

    N_SAMPLES = 10  # 10 sample per encoding
    ENCODINGS = ["angle", "dense", "iqp", "reupload"]

    # Load test data
    test_X, test_y = load_test_data("imdb", n_samples=N_SAMPLES)
    test_X_scaled = torch.sigmoid(test_X) * math.pi
    print(f"\nTest data: {test_X.shape}, labels: {test_y.tolist()}")

    # Load all encoding models
    models = {}
    params = {}
    for enc in ENCODINGS:
        model, thetas, cw, cb = load_encoding_model(enc)
        if model is not None:
            models[enc] = model
            params[enc] = (thetas, cw, cb)

    if not models:
        print("ERROR: No models loaded!")
        return

    # Simulator inference (reference)
    print("\n" + "=" * 60)
    print("SIMULATOR INFERENCE (reference)")
    print("=" * 60)

    sim_results = {}
    for enc in ENCODINGS:
        if enc not in models:
            continue
        preds, acc = simulator_inference(models[enc], test_X, test_y)
        sim_results[enc] = {"preds": preds.tolist(), "accuracy": acc}
        print(f"  {enc}: {acc:.4f} ({int(acc * N_SAMPLES)}/{N_SAMPLES})")

    # IBM QPU inference
    print("\n" + "=" * 60)
    print("IBM QPU INFERENCE")
    print("=" * 60)

    from qiskit_ibm_runtime import QiskitRuntimeService

    print("\n  IBM'e baglaniliyor...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.least_busy(operational=True, min_num_qubits=8)
    print(f"  Backend: {backend.name} ({backend.num_qubits} qubit)")

    qpu_results = {}
    total_qpu_time = 0

    for enc in ENCODINGS:
        if enc not in params:
            continue
        thetas, cw, cb = params[enc]
        try:
            preds, acc, dt = run_qpu_inference(
                enc, thetas, cw, cb,
                test_X_scaled, test_y, backend,
                n_samples=N_SAMPLES
            )
            qpu_results[enc] = {
                "preds": preds.tolist(),
                "accuracy": acc,
                "qpu_time": dt,
            }
            total_qpu_time += dt
        except Exception as e:
            print(f"  ERROR ({enc}): {e}")
            import traceback
            traceback.print_exc()

    # Comparison table
    print("\n" + "=" * 60)
    print("KARSILASTIRMA: Simulator vs IBM QPU")
    print("=" * 60)
    print(f"\n{'Encoding':<15} | {'Sim Acc':>8} | {'QPU Acc':>8} | {'Fark':>6} | {'Noise Gap':>10}")
    print("-" * 60)

    all_results = {"n_samples": N_SAMPLES, "backend": backend.name, "encodings": {}}

    for enc in ENCODINGS:
        if enc not in sim_results or enc not in qpu_results:
            continue
        sim_acc = sim_results[enc]["accuracy"]
        qpu_acc = qpu_results[enc]["accuracy"]
        gap = sim_acc - qpu_acc
        all_results["encodings"][enc] = {
            "sim_accuracy": sim_acc,
            "qpu_accuracy": qpu_acc,
            "noise_gap": gap,
            "qpu_time": qpu_results[enc]["qpu_time"],
            "sim_preds": sim_results[enc]["preds"],
            "qpu_preds": qpu_results[enc]["preds"],
        }
        print(f"{enc:<15} | {sim_acc:>7.1%} | {qpu_acc:>7.1%} | {gap:>+5.1%} | "
              f"{'LOW' if abs(gap) < 0.1 else 'MEDIUM' if abs(gap) < 0.2 else 'HIGH'}")

    all_results["total_qpu_time"] = total_qpu_time

    # Save results
    results_path = PROJECT_ROOT / "results" / "logs" / "ibm_encoding_comparison.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSonuclar: {results_path}")
    print(f"Toplam QPU suresi: {total_qpu_time:.1f}s ({total_qpu_time/60:.1f} dk)")


if __name__ == "__main__":
    main()
