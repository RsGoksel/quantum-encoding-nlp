# Quantum Encoding Selection for NLP Sentiment Analysis

A systematic comparison of four variational quantum circuit (VQC) data-encoding strategies — **Angle**, **Dense Angle**, **IQP**, and **Data Re-uploading** — for binary sentiment classification, with multi-seed validation on IMDb / SST-2 and real-hardware verification on IBM Quantum (`ibm_fez`, 156 qubits).

> Companion code for the paper *"Evaluating Quantum Data Encoding Strategies and Gradient Trainability for NLP Sentiment Analysis"* (Gündüz, 2026). Draft: [`docs/paper_draft.md`](docs/paper_draft.md).

---

## Highlights

- **Dataset-agnostic encoding ranking.** Angle > Dense > IQP > Re-uploading — preserved across IMDb (5 seeds) and SST-2.
- **Gradient variance is informative but not deterministic.** High variance helps optimization but does not guarantee accuracy (Re-uploading: variance 9.10 → accuracy 60%); low variance does not preclude strong performance (Dense Angle: 0.86 → 67.6%).
- **Sigmoid scaling is a +17 pp design lever.** Aligning PCA outputs (~N(0,1)) to the encoder domain `[0, π]` is a first-order concern, not a hyperparameter footnote.
- **Hardware validation with T1 bias control.** A class-balanced 10-sample test on `ibm_fez` reveals 100% simulator–QPU prediction agreement with symmetric per-class fidelity — no T1 amplitude-damping confound at the scales tested.
- **3,400× faster training** via PyTorch-native statevector simulation with autograd, vs. Qiskit's parameter-shift rule (14 ms vs. 1,501 ms per backward pass).

## Pipeline

```
Text  ──DistilBERT (frozen)──► R^768 ──PCA──► R^8 ──σ(x)·π──► [0,π]^8
                                                                  │
                                            ┌─────────────────────┘
                                            ▼
                              Parameterized Quantum Circuit (8 qubits, reps=1)
                                            │
                                            ▼
                              ⟨Z⟩ per qubit  ──Linear──►  logits ∈ R^2
```

## Headline Results

### Encoding comparison — IMDb (5 seeds) + SST-2 (seed 42)

| Encoding     | Qubits | Params | IMDb (mean ± std) | SST-2  | Grad Var |
|--------------|:------:|:------:|:-----------------:|:------:|:--------:|
| **Angle**    | 8      | 50     | **70.1 ± 1.1 %**  | **76.4 %** | 9.757 |
| Dense Angle  | 4      | 26     | 67.6 ± 0.5 %      | 73.0 % | 0.855    |
| IQP          | 8      | 50     | 62.7 ± 0.6 %      | 66.5 % | 0.899    |
| Re-uploading*| 8      | 50     | 60.2 ± 0.9 %      | 61.1 % | 9.098    |
| *Linear baseline* | — | *18*  | *72.2 %*          | —      | —        |

\* Constrained variant (no entangling layer after re-upload — see `docs/paper_draft.md` §5.2).

### IBM Quantum hardware — `ibm_fez` (Eagle r3, 156 qubits), balanced 5+5 test set

| Encoding | Sim Acc | QPU Acc | Sim–QPU Agreement | L0 Agree | L1 Agree | Transpiled Depth |
|----------|:-------:|:-------:|:-----------------:|:--------:|:--------:|:----------------:|
| Angle    | 80 %    | 80 %    | **10/10 (100 %)**  | 5/5      | 5/5      | 19               |
| Dense    | 80 %    | 80 %    | **10/10 (100 %)**  | 5/5      | 5/5      | 18               |
| IQP      | 80 %    | 80 %    | **10/10 (100 %)**  | 5/5      | 5/5      | **101**          |
| Reupload | 80 %    | 80 %    | **10/10 (100 %)**  | 5/5      | 5/5      | 21               |

Symmetric per-class agreement rules out T1 amplitude-damping bias.

## Repository Layout

```
quantum-attention/
├── data/
│   ├── preprocess.py             # DistilBERT embedding + PCA fitting
│   ├── imdb_embeddings.pt        # cached 8-dim PCA features (IMDb)
│   └── sst2_embeddings.pt        # cached 8-dim PCA features (SST-2)
├── models/
│   ├── classical_baseline.py     # Linear(8,2)
│   ├── classical_attention.py    # Q/K/V baseline
│   └── quantum_encodings.py      # Angle / Dense / IQP / Re-uploading + ansatz
├── training/
│   ├── config.py                 # TrainingConfig dataclass
│   └── train.py                  # unified training loop (all model types)
├── experiments/
│   ├── run_multiseed.py          # batch runner (skip-if-exists)
│   ├── barren_plateau.py         # gradient-variance analysis (100 random inits)
│   ├── ibm_encoding_inference.py # QPU inference: 4 encodings × 10 samples
│   ├── ibm_balanced_qpu.py       # T1-controlled balanced (5+5) QPU run
│   └── paper_analysis.py         # reproduces all paper figures
├── results/
│   ├── logs/                     # per-experiment JSON history files
│   └── plots/                    # paper figures (fig1 … fig5)
├── checkpoints/                  # trained .pt files (one per (model, dataset, seed))
├── docs/
│   ├── paper_draft.md            # full paper draft
│   ├── paper.tex                 # LaTeX source
│   └── plans/                    # design + implementation plans
├── CLAUDE.md                     # project context for LLM sessions
├── system_map.md                 # detailed system map
├── developments.md               # chronological dev log
└── ibm_quantum_almanac.md        # IBM Quantum API notes & gotchas
```

## Setup

```bash
git clone https://github.com/RsGoksel/quantum-encoding-nlp.git
cd quantum-encoding-nlp
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Tested on Python 3.13, PyTorch 2.6 (CUDA 12.4), Qiskit 2.3, qiskit-ibm-runtime 0.45.

### Optional: IBM Quantum access

Only needed to reproduce hardware experiments — simulator-only training works without it.

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",   # NOT 'ibm_quantum' — Qiskit ≥ 2.x
    token="YOUR_IBM_QUANTUM_TOKEN",   # https://quantum.ibm.com/account
    overwrite=True,
)
```

## Reproducing Paper Results

The cached embeddings (`data/*.pt`) are committed, so step (1) is optional.

```bash
# (1) optional — re-embed text with DistilBERT and refit PCA
python data/preprocess.py

# (2) train all (encoding × seed) combinations, skip-if-exists
python experiments/run_multiseed.py --dataset imdb --seeds 42 123 456 789 2024
python experiments/run_multiseed.py --dataset sst2 --seeds 42

# (3) gradient-variance analysis (qubit + depth scaling)
python experiments/barren_plateau.py

# (4) IBM Quantum hardware inference (requires IBM account)
python experiments/ibm_balanced_qpu.py        # primary balanced 5+5 run
python experiments/ibm_encoding_inference.py  # initial 4×10 (all label 0)

# (5) regenerate every paper figure + summary table
python experiments/paper_analysis.py
```

> Windows users: prefix Python invocations with `PYTHONIOENCODING=utf-8` to avoid Unicode console errors.

### Train one encoding manually

```bash
python training/train.py \
  --model enc_angle \
  --dataset imdb \
  --reps 1 \
  --lr_quantum 0.05 \
  --subset \
  --seed 42 \
  --suffix _seed42
```

`--model` accepts: `baseline`, `attention`, `quantum`, `enc_{angle,dense,iqp,reupload}`, with optional `_rand` suffix for random initialization.

## Implementation Notes

- **Quantum simulation.** PyTorch-native statevector (256 complex amplitudes for 8 qubits) on CPU, with autograd. ~3,400× faster than the parameter-shift rule for our circuits.
- **Variational ansatz** (shared by all encodings): `Ry · Rz → CZ (linear) → Ry · Rz`, `reps=1`, identity initialization (θ = 0) per [Mele et al., 2024].
- **Critical bug we fixed.** Initial Qiskit circuits used `EfficientSU2` with the default CX entanglement, while training used CZ. We replaced them with hand-built CZ circuits and cross-validated against PyTorch to 6 decimal places (see `docs/paper_draft.md` Appendix A).
- **Sigmoid scaling.** PCA features `~ N(0,1)` are mapped via `σ(x)·π ∈ [0,π]` to cover the full Bloch arc. Without it, accuracy collapses to ~52 % (majority class).

## Limitations

We make **no claim of quantum advantage**. The 8-qubit circuits are classically simulable, the linear baseline outperforms all VQCs, and PCA retains only ~36 % of the DistilBERT variance. The contribution is methodological: a controlled comparison + gradient-trainability diagnostic + hardware fidelity protocol for selecting an encoding within the quantum design space. See §5.4 of the paper for the full limitations list.

## Citation

```bibtex
@misc{gunduz2026quantumencoding,
  author = {G\"und\"uz, Kadir G\"oksel},
  title  = {Evaluating Quantum Data Encoding Strategies and Gradient Trainability
            for NLP Sentiment Analysis},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/RsGoksel/quantum-encoding-nlp},
}
```

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

IBM Quantum Open Plan (10 min/month) for free QPU access; HuggingFace for DistilBERT and the IMDb / SST-2 dataset cards; the Qiskit and PyTorch communities.
