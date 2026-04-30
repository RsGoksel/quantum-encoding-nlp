"""
IBM QPU Balanced Inference — T1 Relaxation Bias Test
=====================================================
Dengeli test seti (5 Label 0 + 5 Label 1) ile IBM QPU inference.

AMAÇ: T1 relaxation bias'ini tespit etmek.
  - T1 varsa: QPU Label 1 tahminlerinde sistematik kayma (0'a dogru)
  - T1 yoksa: Angle encoding hem L0 hem L1 icin yuksek fidelity → guclu claim

OPTİMİZASYON (eski script'e gore):
  - batch_size=10 (eski: 5) → 4 job yerine min 4 job ama her biri daha verimli
  - IQP ve Reupload icin ayni batch_size, Dense icin daha az PUB (4 qubit)
  - Her encoding tek job ile tamamlanir (10 sample x 8 qubit = 80 PUB/job)
  - Toplam tahmini sure: ~5-6 dakika

KULLANIM:
    PYTHONIOENCODING=utf-8 python experiments/ibm_balanced_qpu.py
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
# 1. Dengeli test seti
# ===========================================================================

def load_balanced_test_data(dataset="imdb", n_per_class=5, seed=42):
    """5 Label 0 + 5 Label 1 reproducible balanced subset."""
    data = torch.load(PROJECT_ROOT / "data" / f"{dataset}_embeddings.pt", weights_only=False)
    test_X = data["test_X"]
    test_y = data["test_y"]

    label_0_idx = (test_y == 0).nonzero(as_tuple=True)[0]
    label_1_idx = (test_y == 1).nonzero(as_tuple=True)[0]

    rng = np.random.RandomState(seed)
    sel_0 = label_0_idx[rng.choice(len(label_0_idx), n_per_class, replace=False)]
    sel_1 = label_1_idx[rng.choice(len(label_1_idx), n_per_class, replace=False)]

    # Label 0 once, Label 1 sonra (indices: 0..4 = L0, 5..9 = L1)
    selected = torch.cat([sel_0, sel_1])
    balanced_X = test_X[selected]
    balanced_y = test_y[selected]

    print(f"  Test seti: {len(balanced_X)} sample ({n_per_class} x Label 0 + {n_per_class} x Label 1)")
    print(f"  Label 0 data indeksleri: {sel_0.tolist()}")
    print(f"  Label 1 data indeksleri: {sel_1.tolist()}")
    print(f"  Labels: {balanced_y.tolist()}")
    return balanced_X, balanced_y


# ===========================================================================
# 2. Model yukleme
# ===========================================================================

def load_encoding_model(encoding_name, dataset="imdb"):
    from models.quantum_encodings import QuantumEncodingModel

    enc_map = {
        "angle":    ("angle",       8),
        "dense":    ("dense_angle", 8),
        "iqp":      ("iqp",         8),
        "reupload": ("re-uploading",8),
    }
    enc_type, n_qubits = enc_map[encoding_name]

    model = QuantumEncodingModel(encoding=enc_type, n_qubits=n_qubits, reps=1, identity_init=True)

    ckpt_path = PROJECT_ROOT / "checkpoints" / f"enc_{encoding_name}_{dataset}_best.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / f"enc_{encoding_name}_{dataset}_seed42_best.pt"
    if not ckpt_path.exists():
        print(f"  WARNING: {ckpt_path} bulunamadi")
        return None, None, None, None

    ckpt = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    thetas     = model.circuit.thetas.detach().numpy()
    clf_w      = model.classifier.weight.detach().numpy()
    clf_b      = model.classifier.bias.detach().numpy()
    n_q_actual = model.circuit.n_qubits
    print(f"  {encoding_name}: val_acc={ckpt['val_acc']:.3f}, {n_q_actual}q, {len(thetas)} variational params")

    return model, thetas, clf_w, clf_b


# ===========================================================================
# 3. Simulator gold standard
# ===========================================================================

def simulator_inference(model, test_X, test_y):
    model.eval()
    with torch.no_grad():
        logits = model(test_X)
        preds  = logits.argmax(dim=1)
    return preds.numpy()


# ===========================================================================
# 4. Qiskit circuit builders (PyTorch ile birebir esit)
# ===========================================================================

def _variational_layers(qc, n_qubits, reps, var_params):
    """Ry → Rz → CZ (son rep'te CZ yok) — PyTorch QuantumCircuitBase ile ayni."""
    idx = 0
    for rep in range(reps + 1):
        for q in range(n_qubits):
            qc.ry(var_params[idx], q); idx += 1
        for q in range(n_qubits):
            qc.rz(var_params[idx], q); idx += 1
        if rep < reps:
            for q in range(n_qubits - 1):
                qc.cz(q, q + 1)
    return idx


def build_angle_circuit(n_qubits=8, reps=1):
    from qiskit.circuit import QuantumCircuit, ParameterVector
    x = ParameterVector("x", n_qubits)
    t = ParameterVector("t", n_qubits * 2 * (reps + 1))
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(x[i], i)
    _variational_layers(qc, n_qubits, reps, t)
    return qc, x, t, None


def build_dense_angle_circuit(n_features=8, reps=1):
    from qiskit.circuit import QuantumCircuit, ParameterVector
    n_q = n_features // 2
    x = ParameterVector("x", n_features)
    t = ParameterVector("t", n_q * 2 * (reps + 1))
    qc = QuantumCircuit(n_q)
    for q in range(n_q):
        qc.ry(x[2 * q], q)
        qc.rz(x[2 * q + 1], q)
    _variational_layers(qc, n_q, reps, t)
    return qc, x, t, None


def build_iqp_circuit(n_qubits=8, reps=1):
    from qiskit.circuit import QuantumCircuit, ParameterVector
    x   = ParameterVector("x",  n_qubits)
    xx  = ParameterVector("xx", n_qubits - 1)
    t   = ParameterVector("t",  n_qubits * 2 * (reps + 1))
    qc  = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
    for q in range(n_qubits):
        qc.rz(x[q], q)
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
        qc.rz(xx[q], q + 1)
        qc.cx(q, q + 1)
    _variational_layers(qc, n_qubits, reps, t)
    return qc, x, t, xx


def build_reupload_circuit(n_qubits=8, reps=1):
    from qiskit.circuit import QuantumCircuit, ParameterVector
    x   = ParameterVector("x", n_qubits)
    t   = ParameterVector("t", n_qubits * 2 * (reps + 1))
    qc  = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.ry(x[q], q)
    idx = 0
    for rep in range(reps + 1):
        for q in range(n_qubits):
            qc.ry(t[idx], q); idx += 1
        for q in range(n_qubits):
            qc.rz(t[idx], q); idx += 1
        if rep < reps:
            for q in range(n_qubits - 1):
                qc.cz(q, q + 1)
            for q in range(n_qubits):
                qc.ry(x[q], q)
    return qc, x, t, None


BUILDERS = {
    "angle":    (build_angle_circuit,       8),
    "dense":    (build_dense_angle_circuit, 4),
    "iqp":      (build_iqp_circuit,         8),
    "reupload": (build_reupload_circuit,    8),
}


# ===========================================================================
# 5. IBM QPU inference — tek encoding, TUM sample'lar tek job'da
# ===========================================================================

def run_qpu_inference(enc_name, thetas, clf_w, clf_b,
                      test_X_scaled, test_y, backend, n_samples=10):
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2

    build_fn, n_qubits = BUILDERS[enc_name]
    qc, x_params, t_params, xx_params = build_fn()
    print(f"\n  [{enc_name.upper()}] {n_qubits} qubit, {n_samples} sample")

    # Observables
    obs = [
        SparsePauliOp.from_list([("I"*(n_qubits-1-i) + "Z" + "I"*i, 1.0)])
        for i in range(n_qubits)
    ]

    # Transpile (optimization_level=1 — hizli ve iyi)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    tc = pm.run(qc)
    t_obs = [o.apply_layout(tc.layout) for o in obs]
    depth = tc.depth()
    print(f"  Transpiled depth: {depth}")

    data_np = test_X_scaled[:n_samples].numpy()

    # --- OPTIMIZE: Tum sample'lari tek job'da gonder ---
    # 10 sample x 8 qubit = 80 PUB — IBM Open Plan icin guvenli
    pubs = []
    for i in range(n_samples):
        pv = {}
        for j, p in enumerate(x_params):
            pv[p] = float(data_np[i, j])
        if xx_params is not None:
            for j in range(len(xx_params)):
                pv[xx_params[j]] = float(data_np[i, j] * data_np[i, j+1])
        for j, p in enumerate(t_params):
            pv[p] = float(thetas[j])
        bound = tc.assign_parameters(pv)
        for o in t_obs:
            pubs.append((bound, o))

    estimator = EstimatorV2(mode=backend)
    t0 = time.time()

    # Gonder — eger PUB limiti asarsa 2'ye bol
    try:
        job = estimator.run(pubs)
        print(f"  Job {job.job_id()}: {len(pubs)} PUB, {n_samples} sample tek job'da")
        result = job.result()
        all_z = np.zeros((n_samples, n_qubits))
        for i in range(n_samples):
            for q in range(n_qubits):
                all_z[i, q] = result[i*n_qubits + q].data.evs
    except Exception as e:
        if "6073" in str(e) or "limit" in str(e).lower():
            # Fallback: 5'erli batch
            print(f"  PUB limit — 5'erli batch'e donuluyor...")
            all_z = np.zeros((n_samples, n_qubits))
            for bs in range(0, n_samples, 5):
                be = min(bs+5, n_samples)
                sub_pubs = pubs[bs*n_qubits : be*n_qubits]
                job = estimator.run(sub_pubs)
                print(f"  Job {job.job_id()}: {len(sub_pubs)} PUB, sample {bs}-{be-1}")
                res = job.result()
                for i in range(be-bs):
                    for q in range(n_qubits):
                        all_z[bs+i, q] = res[i*n_qubits+q].data.evs
        else:
            raise

    dt = time.time() - t0

    # Classify
    logits = all_z @ clf_w.T + clf_b
    preds  = np.argmax(logits, axis=1)
    labels = test_y[:n_samples].numpy()
    acc    = (preds == labels).mean()

    # Per-label accuracy
    l0_mask = labels == 0
    l1_mask = labels == 1
    l0_acc = (preds[l0_mask] == 0).mean() if l0_mask.sum() > 0 else float("nan")
    l1_acc = (preds[l1_mask] == 1).mean() if l1_mask.sum() > 0 else float("nan")

    print(f"  QPU Acc: {acc:.1%} | Label0: {l0_acc:.1%} ({int(l0_acc*l0_mask.sum())}/{l0_mask.sum()}) | "
          f"Label1: {l1_acc:.1%} ({int(l1_acc*l1_mask.sum()) if not np.isnan(l1_acc) else 0}/{l1_mask.sum()}) | {dt:.1f}s")

    return preds, float(acc), dt, depth, all_z


# ===========================================================================
# 6. T1 Relaxation Bias Analizi
# ===========================================================================

def t1_bias_analysis(enc_results, test_y_np, n_per_class):
    """
    QPU'nun Label 0 vs Label 1 performansini karsilastir.
    T1 relaxation kubitleri |0>'a ceker → Label 1 tahminleri daha cok etkilenir.
    """
    print("\n" + "="*70)
    print("T1 RELAXATION BIAS ANALIZI")
    print("="*70)
    print(f"\nT1 relaxation kubitleri |0> durumuna ceker.")
    print(f"Kiyaslama: QPU label0 agreement vs QPU label1 agreement\n")

    l0_mask = test_y_np == 0
    l1_mask = test_y_np == 1

    rows = []
    for enc, d in enc_results.items():
        sp = np.array(d["sim_preds"])
        qp = np.array(d["qpu_preds"])

        l0_agree = (sp[l0_mask] == qp[l0_mask]).mean()
        l1_agree = (sp[l1_mask] == qp[l1_mask]).mean()
        overall  = (sp == qp).mean()

        # T1 bias: QPU pred=0 rate > sim pred=0 rate
        shift = (qp == 0).mean() - (sp == 0).mean()

        # Bias flag: QPU dramatik olarak L1'de kotu ama L0'da iyi ise
        if l0_agree - l1_agree >= 0.4:
            flag = "STRONG T1 BIAS"
        elif l0_agree - l1_agree >= 0.2:
            flag = "MODERATE BIAS"
        elif abs(l0_agree - l1_agree) < 0.2:
            flag = "NO BIAS"
        else:
            flag = "L1 > L0 (unexpected)"

        rows.append((enc, l0_agree, l1_agree, overall, shift, flag))
        d["l0_agreement"] = float(l0_agree)
        d["l1_agreement"] = float(l1_agree)
        d["pred0_shift"]  = float(shift)
        d["t1_flag"]      = flag

    header = f"{'Enc':<10} | {'L0 Agr':>8} | {'L1 Agr':>8} | {'Overall':>8} | {'0-Shift':>8} | Verdict"
    print(header)
    print("-"*len(header))
    for enc, l0, l1, ov, sh, flag in rows:
        print(f"{enc:<10} | {l0:>7.0%} | {l1:>7.0%} | {ov:>7.0%} | {sh:>+7.0%} | {flag}")

    return rows


# ===========================================================================
# 7. Main
# ===========================================================================

def main():
    print("="*70)
    print("IBM QPU BALANCED INFERENCE — T1 Relaxation Bias Test")
    print(f"Tarih: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    N_PER_CLASS = 5
    N_SAMPLES   = N_PER_CLASS * 2   # 10
    ENCODINGS   = ["angle", "dense", "iqp", "reupload"]

    # ----- 1. Dengeli test seti -----
    print("\n[1/4] DENGELI TEST SETI OLUSTURULUYOR")
    test_X, test_y = load_balanced_test_data("imdb", n_per_class=N_PER_CLASS, seed=42)
    test_X_scaled = torch.sigmoid(test_X) * math.pi
    print(f"  Scaled: [{test_X_scaled.min():.3f}, {test_X_scaled.max():.3f}]")

    # ----- 2. Model yukleme -----
    print("\n[2/4] MODEL YUKLEME")
    models, params = {}, {}
    for enc in ENCODINGS:
        m, th, cw, cb = load_encoding_model(enc)
        if m:
            models[enc] = m
            params[enc] = (th, cw, cb)

    # ----- 3. Simulator gold standard -----
    print("\n" + "="*70)
    print("[3/4] SIMULATOR GOLD STANDARD")
    print("="*70)
    test_y_np = test_y.numpy()
    l0_mask   = test_y_np == 0
    l1_mask   = test_y_np == 1

    sim_results = {}
    for enc in ENCODINGS:
        if enc not in models:
            continue
        sp  = simulator_inference(models[enc], test_X, test_y)
        l0a = (sp[l0_mask] == 0).mean()
        l1a = (sp[l1_mask] == 1).mean()
        acc = (sp == test_y_np).mean()
        sim_results[enc] = {"preds": sp.tolist(), "accuracy": float(acc)}
        print(f"  {enc:<10}: {acc:.1%} overall | L0: {l0a:.1%} | L1: {l1a:.1%} | preds: {sp.tolist()}")

    # ----- 4. IBM QPU inference -----
    print("\n" + "="*70)
    print("[4/4] IBM QPU INFERENCE")
    print("="*70)

    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.least_busy(operational=True, min_num_qubits=8)
    print(f"\n  Backend: {backend.name} ({backend.num_qubits} qubit)")

    qpu_results   = {}
    total_qpu_t   = 0
    enc_data_all  = {}

    for enc in ENCODINGS:
        if enc not in params:
            continue
        th, cw, cb = params[enc]
        try:
            preds, acc, dt, depth, z_vals = run_qpu_inference(
                enc, th, cw, cb, test_X_scaled, test_y, backend, N_SAMPLES
            )
            qpu_results[enc] = {
                "preds":           preds.tolist(),
                "accuracy":        acc,
                "qpu_time":        dt,
                "transpiled_depth":depth,
                "z_expectations":  z_vals.tolist(),
            }
            total_qpu_t += dt
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR ({enc}): {e}")

    # ----- Comparison table -----
    print("\n" + "="*70)
    print("KARSILASTIRMA: Simulator vs IBM QPU (DENGELI SET)")
    print("="*70)
    print(f"\n{'Enc':<10} | {'Sim':>6} | {'QPU':>6} | {'Gap':>6} | {'Agree':>7} | {'Depth':>5}")
    print("-"*55)

    for enc in ENCODINGS:
        if enc not in sim_results or enc not in qpu_results:
            continue
        sa  = sim_results[enc]["accuracy"]
        qa  = qpu_results[enc]["accuracy"]
        sp  = np.array(sim_results[enc]["preds"])
        qp  = np.array(qpu_results[enc]["preds"])
        agr = (sp == qp).mean()
        dep = qpu_results[enc]["transpiled_depth"]
        enc_data_all[enc] = {
            **qpu_results[enc],
            "sim_accuracy": sa,
            "noise_gap":    sa - qa,
            "agreement":    float(agr),
            "sim_preds":    sim_results[enc]["preds"],
        }
        print(f"{enc:<10} | {sa:>5.1%} | {qa:>5.1%} | {sa-qa:>+5.1%} | {agr:>6.0%} | {dep:>5}")

    # ----- T1 Bias analizi -----
    t1_rows = t1_bias_analysis(enc_data_all, test_y_np, N_PER_CLASS)

    # ----- Sample-level detail -----
    print("\n" + "="*70)
    print("SAMPLE-LEVEL DETAY")
    print("="*70)
    labels = test_y.tolist()
    header = f"{'#':>2} | {'Lbl':>3} | "
    for enc in ENCODINGS:
        if enc in enc_data_all:
            header += f"{'Sim':>3}{'QPU':>4}{'OK?':>5} | "
    print(header)
    print("-"*80)
    for i in range(N_SAMPLES):
        row = f"{i:>2} | {labels[i]:>3} | "
        for enc in ENCODINGS:
            if enc not in enc_data_all:
                continue
            sp = enc_data_all[enc]["sim_preds"][i]
            qp = enc_data_all[enc]["qpu_preds"][i]
            c  = "OK" if sp == qp else "DIFF"
            sv = "v" if sp == labels[i] else "x"
            qv = "v" if qp == labels[i] else "x"
            row += f" {sp}({sv}) {qp}({qv}) {c:>4} | "
        print(row)
    print(f"\n  [v]=correct, [x]=wrong, OK=sim==qpu match, DIFF=diverged")

    # ----- Save -----
    out = {
        "experiment":     "balanced_t1_bias_test",
        "date":           time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_per_class":    N_PER_CLASS,
        "n_samples":      N_SAMPLES,
        "backend":        backend.name,
        "labels":         labels,
        "total_qpu_time": total_qpu_t,
        "encodings":      enc_data_all,
    }
    out_path = PROJECT_ROOT / "results" / "logs" / "ibm_balanced_qpu_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSonuclar kaydedildi: {out_path}")
    print(f"Toplam QPU suresi: {total_qpu_t:.1f}s ({total_qpu_t/60:.1f} dk)")

    # ----- Final verdict -----
    print("\n" + "="*70)
    print("FINAL: T1 RELAXATION BIAS VERDICT")
    print("="*70)
    for enc, l0, l1, ov, sh, flag in t1_rows:
        star = "  ★" if flag == "NO BIAS" else ("  !" if "STRONG" in flag else "")
        print(f"  {enc:<10}: L0={l0:.0%}, L1={l1:.0%}, shift={sh:+.0%} → {flag}{star}")

    any_bias = any("BIAS" in r[5] for r in t1_rows)
    no_bias  = all("NO BIAS" in r[5] for r in t1_rows)
    if no_bias:
        print("\n  SONUC: T1 bias yok. QPU hem L0 hem L1'de esit fidelity gosteriyor.")
        print("  → Angle encoding'in noise resilience iddiasi GUCLU temele sahip.")
    elif any_bias:
        print("\n  SONUC: T1 bias tespit edildi. Bazi encoding'lerde L1 fidelity daha dusuk.")
        print("  → Paper'da bu sinirlamadan donusturucu bicimde bahsedilmeli.")
    else:
        print("\n  SONUC: Karisik bulgular. Detaylari yukari bak.")


if __name__ == "__main__":
    main()
