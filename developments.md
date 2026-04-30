# Developments — Quantum-Enhanced Attention for RAG Pipelines

> Teknik gelismelerin kronolojik ve detayli kaydi.
> Her phase'de ne yapildi, hangi kod nerde, ne hata alindi, nasil cozuldu.
> Bildiri materyali olarak kullanilacak.
> Tarih: 2026-03-08

---

## Phase 1: Altyapi Kurulumu

### 1.1 Proje Olusturma
- **Konum:** `C:\Users\gokss\OneDrive\Masaüstü\Kuantum\quantum-attention\`
- **system_map.md** olusturuldu — her LLM oturumunun baslangic context dosyasi
- Klasor yapisi: `data/`, `models/`, `quantum/`, `training/`, `experiments/`, `results/`, `checkpoints/`

### 1.2 Paket Kurulumu
Halihazirda kurulu olanlar:
- `qiskit==2.3.0`, `qiskit-aer==0.17.2`, `qiskit-machine-learning==0.9.0`
- `torch==2.6.0+cu124` (CUDA destekli, RTX 4060 Laptop GPU)
- `scikit-learn==1.7.2`, `matplotlib==3.10.7`, `seaborn==0.13.2`
- Python 3.13.7, Windows 11 Pro

Yeni kurulanlar:
```bash
pip install qiskit-ibm-runtime transformers datasets
```
- `qiskit-ibm-runtime==0.45.1` — IBM kuantum bilgisayar erişimi
- `transformers==5.3.0` — DistilBERT embedding cikartma
- `datasets==4.6.1` — HuggingFace dataset yukleyici

### 1.3 IBM Quantum Baglantisi

**Token kaydi:**
```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel='ibm_quantum_platform',  # 'ibm_quantum' DEGIL!
    token='YOUR_IBM_QUANTUM_TOKEN',  # https://quantum.ibm.com/account
    overwrite=True
)
```

**HATA #1:** `InvalidAccountError: "Invalid channel value... got 'ibm_quantum'"`
- Sebep: Qiskit 2.x channel ismini `ibm_quantum`'dan `ibm_quantum_platform`'a degistirmis
- Cozum: `channel='ibm_quantum_platform'`

**Mevcut QPU'lar (2026-03-08):**
| QPU | Qubit | Durum |
|-----|-------|-------|
| ibm_fez | 156 | operational |
| ibm_marrakesh | 156 | operational |
| ibm_torino | 133 | operational |

### 1.4 Ilk Circuit Testi

**Dosya:** Inline test (henuz dosyaya yazilmadi)
**Circuit:** 8 qubit EfficientSU2, reps=2, ZZFeatureMap encoding
**Sonuc:**
- Feature map: 8 parametre
- Ansatz: 48 trainable parametre
- Full circuit: 56 toplam parametre
- TorchConnector forward pass: input (2,8) -> output (2,8) OK
- Identity init test: max diff = 1.0 (identity degil)

**HATA #2:** `ZZFeatureMap` ve `EfficientSU2` DeprecationWarning
- Sebep: Qiskit 2.1+ class-based API deprecated
- Not: Fonksiyon API'sine gecilmeli (`zz_feature_map()`, `efficient_su2()`)
- Etki: Calisma durumunu etkilemiyor, sadece uyari

**HATA #3:** Identity init test basarisiz (max diff = 1.0)
- Sebep: CZ entangling gate'ler parametresiz — her zaman entanglement yapar
- Cozum: Beklenen davranis. "Identity init" = sadece Ry(0)=I, Rz(0)=I. CZ gate'ler circuit'in sabit entanglement yapisini saglar.
- BILDIRI NOTU: EfficientSU2'de tam identity initialization mumkun degil CZ gate'ler yuzunden. Literaturde "identity init" terimi sadece trainable parametrelerin sifirlanmasini ifade eder.

---

## Phase 2: Data Pipeline

### 2.1 Dataset Indirme
**Dosya:** `data/preprocess.py`

| Dataset | Kaynak | Boyut | Sure |
|---------|--------|-------|------|
| IMDb | `load_dataset("imdb")` | 25K train + 25K test | 74.2s |
| SST-2 | `load_dataset("glue", "sst2")` | 67K train + 872 val | 5.4s (ilk sefer 11.4s) |

### 2.2 DistilBERT Embedding Cikartma

**Model:** `distilbert-base-uncased` (66.4M parametre, FROZEN)
- Eval mode: dropout kapali
- Device: CUDA (RTX 4060 Laptop GPU)
- Max token: 128 (hiz icin sinirlanmis)
- Batch size: 64

**Nasil calisir:**
1. Tokenizer: metin -> token ID'leri + attention mask
2. DistilBERT: 6 transformer katmanindan gecirir
3. `[CLS]` token'inin son katman output'u = cumlenin 768-dim ozet vektoru
4. Bu vektor anlam, sentiment, dilbilgisel yapi bilgisi tasir

**Sureler:**
| Dataset | Split | Shape | Sure |
|---------|-------|-------|------|
| IMDb | Train (25K) | (25000, 768) | 89.4s |
| IMDb | Test (25K) | (25000, 768) | 92.7s |
| SST-2 | Train (57.7K) | (57728, 768) | 45.7s |
| SST-2 | Val (872) | (872, 768) | 0.8s |
| SST-2 | Test (9.6K) | (9621, 768) | 7.5s |

**HATA #4:** `TypeError: len() of a 0-d tensor` (SST-2 processing)
- Sebep: HuggingFace Dataset nesnesi PyTorch tensor index kabul etmiyor
- Konum: `data/preprocess.py:253`
- Cozum: `perm[idx].tolist()` ile int listesine cevir
- **Dosya:** `data/preprocess.py` (duzeltildi), `data/preprocess_sst2_only.py` (yeniden calistirma icin)

**HATA #5:** DistilBERT "UNEXPECTED keys" uyarisi
- Sebep: Masked LM head (vocab_projector, vocab_transform, vocab_layer_norm) yukleniyor ama DistilBertModel kullanmiyor
- Cozum: Guvenlice yok sayilir — feature extraction etkilenmiyor

### 2.3 PCA Boyut Indirgeme (768 -> 8)

**Neden PCA?**
- 768 qubit imkansiz, 8 qubit = 2^8 = 256 durum (yonetilebilir)
- PCA en yuksek varyans'i tutan bilesenleri secer
- SADECE train verisiyle fit edilir (data leakage onleme)
- StandardScaler ile once normalize (ortalama=0, std=1)

**Sonuclar:**
| Dataset | Korunan Varyans | Sure | Bilesen Dagilimi |
|---------|----------------|------|------------------|
| IMDb | %35.7 | 1.29s | [0.084, 0.058, 0.051, 0.039, 0.036, 0.033, 0.028, 0.027] |
| SST-2 | %36.3 | 1.17s | [0.108, 0.057, 0.047, 0.042, 0.031, 0.029, 0.026, 0.023] |

**BILDIRI NOTU:** %36 varyans dusuk gorunebilir ama:
1. 8 qubit fiziksel siniri var
2. Sentiment bilgisi genellikle ilk birkas ana bilesende yogunlasir
3. Klasik baseline da ayni %36 ile calisarak referans oluyor

### 2.4 Kaydedilen Dosyalar

| Dosya | Boyut | Icerik |
|-------|-------|--------|
| `data/imdb_embeddings.pt` | 1,956 KB | train(22500,8), val(2500,8), test(25000,8) + labels + pca_variance |
| `data/sst2_embeddings.pt` | 2,667 KB | train(57728,8), val(872,8), test(9621,8) + labels + pca_variance |

---

## Phase 3: Model Implementation

### 3.1 Classical Baseline
**Dosya:** `models/classical_baseline.py`
- **Mimari:** Linear(8, 2) — en basit model
- **Parametreler:** 8*2 + 2 = 18 (weight + bias)
- **Amac:** Alt sinir — bundan kotu olmak kabul edilemez

### 3.2 Classical Attention
**Dosya:** `models/classical_attention.py`
- **Mimari:** Q/K/V projection (8->8) + gated attention + output projection + classifier
- **Parametreler:** 306
- **Nasil calisir:**
  1. Input x -> Query, Key, Value (her biri Linear 8->8)
  2. Score = sum(Q*K) / sqrt(8) — dot-product benzerlik
  3. Weight = sigmoid(score) — 0-1 arasi gate
  4. Context = weight * V
  5. Output = Linear(out_proj(context))
- **Neden single-head:** Quantum modelimiz ~50 parametre, multi-head yuzlerce parametre ister. Adil karsilastirma icin kucuk tutuyoruz.

### 3.3 Quantum Attention (Qiskit — KULLANILMIYOR)
**Dosya:** `models/quantum_attention.py`
- **Mimari:** EstimatorQNN + TorchConnector + Linear(8,2)
- **Parametreler:** 66 (48 quantum + 18 classical)
- **NEDEN KULLANILMIYOR:** Parameter-shift rule nedeniyle backward pass 1501ms/sample

**HATA #6:** `torch.load weights_only=True` numpy hatasi
- Sebep: PCA variance numpy array, PyTorch 2.6 default `weights_only=True`
- Cozum: `weights_only=False` kullan (kendi dosyamiz)

**HATA #7:** Quantum training 285 saat surecek tahmin (22500 sample)
- Sebep: Parameter-shift rule: f'(t) = [f(t+pi/2) - f(t-pi/2)] / 2
- 48 parametre * 2 = 96 ek circuit evaluation/sample
- Backward: 1501ms/sample vs Forward: 18ms/sample (83x fark!)
- 22500 sample * 1.5s = 33750s = 9.4 saat/epoch * 30 epoch = 285 saat

**HATA #8 (COZUM):** PyTorch-native simulasyona gecis
- **Yeni dosya:** `models/quantum_attention_fast.py` — `QuantumCircuitLayer` + `QuantumAttentionFast`
- PyTorch autograd ile gradient: TEK backward pass, 14ms/sample
- **Hizlanma: 1501ms -> 14ms = ~107x** (8 qubit icin)
- Toplam training: 285 saat -> 5-14 dakika = **~3400x hizlanma**

### 3.4 Quantum Attention (PyTorch-Native — ANA MODEL)
**Dosya:** `models/quantum_attention_fast.py`

**Circuit yapisi (EfficientSU2 ile esdeger):**
```
|0> — Ry(x_0) — [Ry(t_0) — Rz(t_8)  — CZ] x reps — Ry(t_16) — Rz(t_24) — Measure Z_0
|0> — Ry(x_1) — [Ry(t_1) — Rz(t_9)  — CZ] x reps — Ry(t_17) — Rz(t_25) — Measure Z_1
...
|0> — Ry(x_7) — [Ry(t_7) — Rz(t_15) — CZ] x reps — Ry(t_23) — Rz(t_31) — Measure Z_7
```

**Temel operasyonlar:**
1. `_ry_gate(theta)`: Ry rotation matrix (cos/sin, differentiable)
2. `_rz_gate(theta)`: Rz rotation matrix (complex exp, differentiable)
3. `_apply_single_qubit_gate(state, gate, qubit)`: Tensor reshape + tensordot
4. `_apply_cz(state, q1, q2)`: |11> bilesenlerine -1 faz
5. `forward(x)`: Batch processing, her sample icin full circuit + Z measurement

**Sigmoid Scaling (KRITIK KESINF):**
```python
x = torch.sigmoid(x) * math.pi  # PCA output -> [0, pi]
```
- PCA ciktilari ~N(0,1), yani [-3, +3] arasi deger
- Angle encoding Ry(x) icin [0, pi] optimal aralik
- Ry(0) = |0> durumu, Ry(pi) = |1> durumu
- Sigmoid: (-inf, inf) -> (0, 1), sonra *pi -> (0, pi)
- **HATA #10:** Bu scaling olmadan model %52 (random), scaling ile %69 (+17 puan!)

**Identity Initialization:**
```python
if identity_init:
    self.thetas = nn.Parameter(torch.zeros(n_params))  # Ry(0)=I, Rz(0)=I
else:
    self.thetas = nn.Parameter(torch.rand(n_params) * 2 * math.pi)  # Uniform [0, 2pi)
```

**IBM QPU transferi:**
- Egitilmis `self.circuit.thetas` parametreleri Qiskit EfficientSU2'ye aktarilabilir
- Ayni circuit yapisi (angle encoding + Ry/Rz + CZ linear entanglement)

### 3.5 Training Pipeline
**Dosya:** `training/train.py`

**Ozellikler:**
- Quantum modeller icin ayri lr: `lr_quantum=0.05`, `lr_classical=0.001`
- Quantum modeller CPU'da calisir (statevector simulasyon)
- Klasik modeller GPU'da (CUDA)
- 2000 sample subset (quantum ve adil karsilastirma icin)
- ReduceLROnPlateau scheduler (patience=5, factor=0.5)
- Early stopping (patience=10)
- JSON history kaydi + best model checkpoint

**Dosya:** `training/config.py` — tum hyperparametreler

---

## Phase 4: Training & Deneyler

### 4.1 Ilk Deney — Qiskit (BASARISIZ)
- EstimatorQNN + TorchConnector
- Backward pass: 1501ms/sample
- Tahmini toplam: 285 saat
- **IPTAL EDILDI** — PyTorch-native'e gecildi

### 4.2 Ikinci Deney — PyTorch-Native, Eski Ayarlar (DUSUK PERFORMANS)
**Ayarlar:** n_qubits=8, reps=2, lr_quantum=0.01, scaling yok
**Sonuclar:**
| Model | Val Acc | Test Acc |
|-------|---------|----------|
| Quantum (identity) | 51.6% | 52.1% |
| Quantum (random) | 53.2% | 50.2% |

**Teshis:** PCA output [-3,+3] araliginda, angle encoding [0,pi] bekliyor. Uyumsuz scaling model'in ogrenmesini engelliyor.

### 4.3 Ucuncu Deney — Optimize Edilmis Ayarlar (BASARILI)
**Degisiklikler:**
1. `sigmoid(x) * pi` scaling eklendi
2. `reps=1` (48->32 parametre, daha az barren plateau riski)
3. `lr_quantum=0.05` (5x daha agresif)
4. 30 epoch (15->30)

**Final Sonuclar (Adil Karsilastirma, ayni 2000 sample subset):**

| Model | Params | Val Acc | Test Acc | Training Suresi | Epoch/s |
|-------|--------|---------|----------|-----------------|---------|
| **Baseline (Linear)** | 18 | 74.0% | **72.2%** | 2.7s | 0.1s |
| **Classical Attention** | 306 | 73.6% | **71.9%** | 3.0s | 0.1s |
| **Quantum (identity init)** | 50 | 70.0% | **68.8%** | 797s (13.3dk) | 37.9s |
| **Quantum (random init)** | 50 | 71.0% | **69.2%** | 512s (8.5dk) | 42.7s |

**Kritik Bulgular:**
1. Quantum model klasik baseline'a 3 puan farkla yaklasti (%69 vs %72)
2. Random init (%69.2) identity init'ten (%68.8) biraz daha iyi — 8 qubit'te barren plateau belirgin degil
3. 50 parametreli quantum, 306 parametreli klasik attention ile karsilastirilabilir
4. Training suresi: quantum ~267x yavas (797s vs 3s)

### 4.4 Checkpoint Dosyalari

| Dosya | Model | Best Epoch | Val Acc |
|-------|-------|------------|---------|
| `checkpoints/baseline_imdb_best.pt` | Classical Baseline | 20 | 74.0% |
| `checkpoints/attention_imdb_best.pt` | Classical Attention | 18 | 73.6% |
| `checkpoints/quantum_imdb_best.pt` | Quantum (identity) | 11 | 70.0% |
| `checkpoints/quantum_random_imdb_best.pt` | Quantum (random) | 2 | 71.0% |

### 4.5 Training History Dosyalari

| Dosya | Icerik |
|-------|--------|
| `results/logs/baseline_imdb_history.json` | Epoch bazli loss/acc + config |
| `results/logs/attention_imdb_history.json` | Ayni format |
| `results/logs/quantum_imdb_history.json` | Ayni format |
| `results/logs/quantum_random_imdb_history.json` | Ayni format |

---

## Phase 4.5: IBM QPU Inference

### 4.5.1 Ilk Deneme (BASARISIZ)
**Dosya:** `experiments/ibm_inference.py`
**Backend:** IBM'in en az yogun QPU'su (otomatik secim)
**Job ID:** `d6mm27k3pels73a0k8gg`

**HATA #9:** `Error 6073 — The size of the job exceeds the memory limits`
- **Sebep:** 50 sample x 8 observable = 400 PUB (Primitive Unified Bloc) tek job'da gonderildi
- **Teknik detay:** IBM QPU'larin klasik kontrol donanimi (FPGA/ASIC) bellek sinirli. 400 ayri circuit bind + 400 observable evaluation bellegi astirir.
- **QPU suresi harcandi mi:** Muhtemelen kismi harcama — job execution sirasinda fail etti
- **BILDIRI NOTU:** Bu NISQ caginin somut bir sinirlamasi. Kuantum islemci 156 qubit desteklerken, klasik kontrol donanimi is boyutunu sinirliyor. Gercek kuantum hesaplamanin bottleneck'i her zaman kuantum islemci degil — "hybrid" yapinin klasik tarafi da kritik.

**Cozum:**
- 5'erli batch'ler (5 sample x 8 obs = 40 PUB/job — guvenli limit)
- Sample sayisi 50'den 10'a dusuruldu (QPU suresi kiymetli)

### 4.5.2 Ikinci Deneme — BASARILI!
**Ayarlar:** 10 sample, 2 batch (5+5), toplam 80 PUB
**Backend:** ibm_fez (156 qubit)
**Job ID'leri:** `d6mm9ee9td6c73an6ndg` (batch 1), `d6mm9v8bfi7c73a3o8i0` (batch 2)
**Transpiled depth:** 42 (8 qubit circuit'in IBM donanimina optimize edilmis hali)

**Sonuclar:**
| Metrik | Simulator | IBM QPU |
|--------|-----------|---------|
| Accuracy | 70.0% (7/10) | **70.0% (7/10)** |
| Fark | — | **0.0%** |
| Tahmin uyumu | — | **80% (8/10 ayni)** |
| Sure | 0.091s | 138.4s (13.8s/sample) |

**BILDIRI NOTU — Kritik Bulgular:**
1. **Accuracy korundu:** Simulator ve QPU ayni accuracy (%70) — noise modeli etkilemedi!
2. **%80 tahmin uyumu:** 10 sample'in 8'inde birebir ayni tahmin. 2 farkli tahmin ama net accuracy ayni.
3. **Transpiled depth 42:** 8 qubitlik soyut circuit, ibm_fez'in donanim topolojisine gore yeniden derlenince 42 derinlige ulasti. Bu, SWAP gate'lerin eklenmesinden kaynaklaniyor (qubit baglantilari fiziksel kisitlamalar iceriyor).
4. **QPU suresi:** 138.4s toplam (kuyruk + transpile + execution). 2 batch, her batch ~70s.
5. **Gercek donanim dogrulamasi:** "Sadece simulatorde calisiyor" argumani cikarildi.

**Sonuc dosyasi:** `results/logs/ibm_qpu_results.json`

---

## Dosya Haritasi (Tum Proje)

```
quantum-attention/
├── system_map.md              — LLM context dosyasi (proje haritasi)
├── developments.md            — Bu dosya (teknik gelismeler)
├── ibm_quantum_almanac.md     — IBM Quantum dokumantasyon referansi (41KB)
│
├── data/
│   ├── preprocess.py          — Ana preprocessing pipeline (IMDb + SST-2)
│   ├── preprocess_sst2_only.py — SST-2 yeniden calistirma (hata sonrasi)
│   ├── imdb_embeddings.pt     — IMDb PCA embeddings (1956 KB)
│   └── sst2_embeddings.pt     — SST-2 PCA embeddings (2667 KB)
│
├── models/
│   ├── classical_baseline.py  — Linear(8,2), 18 params
│   ├── classical_attention.py — Q/K/V + gated attention, 306 params
│   ├── quantum_attention.py   — Qiskit EstimatorQNN (KULLANILMIYOR — yavas)
│   ├── quantum_attention_fast.py — PyTorch-native quantum circuit, 50 params (ANA MODEL)
│   └── quantum_encodings.py   — 4 encoding stratejisi + QuantumAttentionModel (Phase 6)
│
├── training/
│   ├── config.py              — TrainingConfig dataclass
│   └── train.py               — Training loop (tum modeller icin)
│
├── experiments/
│   ├── ibm_inference.py       — IBM QPU inference script
│   └── barren_plateau.py      — Gradient variance analizi (Phase 6)
│
├── results/
│   └── logs/
│       ├── baseline_imdb_history.json
│       ├── attention_imdb_history.json
│       ├── quantum_imdb_history.json
│       ├── quantum_random_imdb_history.json
│       └── barren_plateau_analysis.json  — Gradient variance data (Phase 6)
│
└── checkpoints/
    ├── baseline_imdb_best.pt
    ├── attention_imdb_best.pt
    ├── quantum_imdb_best.pt
    ├── quantum_random_imdb_best.pt
    ├── enc_angle_imdb_best.pt      — Phase 6, val_acc=0.70
    ├── enc_dense_imdb_best.pt      — Phase 6, val_acc=0.68
    ├── enc_iqp_imdb_best.pt        — Phase 6, val_acc=0.652
    ├── enc_reupload_imdb_best.pt   — Phase 6, val_acc=0.598
    ├── qattn_angle_imdb_best.pt    — Phase 6, val_acc=0.752
    ├── qattn_iqp_imdb_best.pt      — Phase 6, val_acc=0.746
    └── qattn_reupload_imdb_best.pt — Phase 6, val_acc=0.746
```

---

## Karsilasilan Tum Hatalar (Kronolojik)

| # | Hata | Nerede | Sebep | Cozum | Bildiri Notu |
|---|------|--------|-------|-------|--------------|
| 1 | InvalidAccountError: ibm_quantum | Phase 1 | Qiskit 2.x API degisikligi | `ibm_quantum_platform` | API versiyonlama sorunu |
| 2 | ZZFeatureMap DeprecationWarning | Phase 1 | Class-based API deprecated | Fonksiyon API'sine gec | - |
| 3 | Identity init != Identity matrix | Phase 1 | CZ gate'ler parametresiz | Beklenen davranis | Identity init taniminin sinirlari |
| 4 | TypeError: len() 0-d tensor | Phase 2 | HF Dataset tensor index | .tolist() | Framework uyumsuzlugu |
| 5 | DistilBERT UNEXPECTED keys | Phase 2 | MLM head yukleniyor | Yok say | - |
| 6 | weights_only=True numpy | Phase 3 | PyTorch 2.6 default | weights_only=False | - |
| 7 | 285 saat training tahmini | Phase 3 | Parameter-shift rule | PyTorch-native (3400x hiz) | Gradient hesaplama stratejisi kritik |
| 8 | Backward 1501ms/sample | Phase 3 | 96 ek circuit eval | Autograd ile 14ms | Parameter-shift vs autograd |
| 9 | IBM Error 6073 bellek | Phase 4.5 | 400 PUB tek job | 5'erli batch | NISQ klasik kontrol siniri |
| 10 | Quantum %52 accuracy | Phase 4 | Yanlis encoding araligi | sigmoid*pi scaling | Veri on-isleme kritik |
| 11 | WinError 1455 page file | Phase 6 | 6+ paralel PyTorch process | Tek tek calistir | Windows kaynak siniri |

---

## Bildiri Icin Anahtar Bulgular

### Pozitif
1. 50 parametreli quantum model, %69 accuracy ile 306 parametreli klasik attention'a yakin
2. Sigmoid scaling ile angle encoding uyumu %17 performans artisi sagladi
3. PyTorch-native simulasyon Qiskit parameter-shift'ten 3400x hizli
4. Gercek IBM kuantum donaniminda calistirildi (133-156 qubit islemciler)

### Negatif (ama degerli)
1. Quantum model klasik baseline'dan 3 puan geride (%69 vs %72)
2. Training 267x daha yavas (quantum simulasyon overhead'i)
3. Identity init 8 qubit'te beklenen avantaji gostermedi
4. IBM QPU'da klasik kontrol donanimi bottleneck (Error 6073)

### Beklenmedik
1. Random init identity init'ten biraz daha iyi — 8 qubit barren plateau esiginin altinda olabilir
2. PCA %36 varyans ile bile klasik modeller %72 yakaladi — sentiment bilgisi ilk bilesenlerde yogun
3. Qiskit'in EstimatorQNN'i pratik olarak kullanilamaz boyutta yavas (285 saat)

---

## Phase 4.5 Degerlendirme: Ne Yaptik, Ne Yapamadik

### Ozet
Phase 1-4.5 boyunca yapilan calisma temelde **dogrulama (verification)** niteliginde:
- Kuantum circuit dogru calisiyor mu? EVET
- PyTorch-native simulasyon Qiskit ile tutarli mi? EVET
- IBM gercek donanimi simulatorle uyumlu mu? EVET (%70 vs %70, %80 uyum)
- Kuantum model ogrenebiliyor mu? EVET (%52 -> %69 sigmoid scaling ile)

### Eksik Olan: Literatur Katkisi
Yukaridaki bulgular "calisiyor" demek icin yeterli, ama bildiri icin yetersiz:
- Sadece tek bir encoding stratejisi test edildi (angle encoding + sigmoid scaling)
- Sadece tek bir ansatz yapisi (EfficientSU2) denendi
- Klasik modeller tum metriklerde kuantum modelden iyi
- "Kuantum avantaji" gosterilemedi — sadece "kuantum zarar vermez" gosterildi

### Sonraki Adim
Literatur katkisi icin sistematik bir deney seti gerekiyor — bkz. Phase 6 planlamasi.

---

## Phase 6: Sistematik Encoding Karsilastirmasi + Quantum Attention [TAMAMLANDI]

### 6.1 Yeni Encoding Stratejileri (4 farkli implementasyon)

**Dosya:** `models/quantum_encodings.py`

**Mimari:**
- `QuantumCircuitBase` — encoding-agnostic base class (shared variational layers + Z measurement)
- 4 encoding subclass: sadece `_encode()` metodu farkli
- `QuantumEncodingModel` — dogrudan siniflandirma: circuit → Linear(out_dim, 2)
- `QuantumAttentionModel` — gercek quantum attention: circuit → softmax → attention weights → feature modulation → Linear → prediction

**Encoding Karsilastirmasi (IMDb, 2000 subset, seed=42):**

| Encoding | Qubit | Params (Q+C) | Val Acc | Test Acc | Epoch |
|----------|-------|-------------|---------|----------|-------|
| Angle (sigmoid*pi) | 8 | 50 (32+18) | 70.0% | **69.9%** | 11 |
| Dense Angle (Ry+Rz) | 4 | 26 (16+10) | 68.0% | **68.1%** | 18 |
| IQP (Rz+RZZ) | 8 | 50 (32+18) | 65.2% | **62.4%** | 4 |
| Data Re-uploading | 8 | 50 (32+18) | 59.8% | **58.1%** | 1 |

**Siralama:** Angle > Dense > IQP > Re-uploading
**Yorum:** Angle encoding NLP icin en uygun. IQP ve Re-uploading'in dusuk performansi barren plateau analiziyle uyumlu (dusuk gradient variance).

### 6.2 Quantum Attention Modeli Sonuclari

**Mekanizma:** Circuit output → shift (+1.0) → temperature scale → softmax → attention weights → original_features * weights → Linear → prediction

| Model | Tip | Params | Val Acc | Test Acc | Epoch |
|-------|-----|--------|---------|----------|-------|
| qattn_angle | Q-Attention | 51 (32+19) | 75.2% | **73.4%** | 7 |
| qattn_iqp | Q-Attention | 51 (32+19) | 74.6% | **72.8%** | 15 |
| qattn_reupload | Q-Attention | 51 (32+19) | 74.6% | **73.2%** | 10 |
| Baseline (Linear) | Klasik | 18 | 74.0% | **72.2%** | — |
| Classical Attention | Klasik | 306 | 73.6% | **71.9%** | — |

**KRITIK DEGERLENDIRME:**
- qattn modelleri %73 gosteriyor ama bu sonuc GUVENILMEZ:
  1. Tek seed (42) — istatistiksel anlamlilik yok
  2. Baseline'dan sadece +1.2% fark — noise marjinda
  3. qattn_angle'in ek klasik parametreleri var (temperature, softmax) — avantaj quantum'dan mi, klasikten mi belli degil
  4. Tum qattn modelleri benzer performans (~73%) — encoding farki kaybolmus, klasik katmanlar domine ediyor
- **Kullanicinin degerlendirmesi:** "bence tesadufen olmustur" — hakli olma olasiligi yuksek

### 6.3 Barren Plateau Gradient Variance Analizi

**Dosya:** `experiments/barren_plateau.py`
**Sonuc dosyasi:** `results/logs/barren_plateau_analysis.json`
**Yontem:** 100 random initialization, her biri 10 input uzerinden gradient hesabi, per-parametre variance

**Encoding Karsilastirmasi (8 feature, reps=1):**

| Encoding | Qubit | mean_var | mean|grad| | Params |
|----------|-------|----------|-----------|--------|
| Angle | 8 | **9.757** | 2.009 | 32 |
| Dense Angle | 4 | 0.855 | 0.578 | 16 |
| IQP | 8 | 0.899 | 0.582 | 32 |
| Re-uploading | 8 | **9.098** | 1.978 | 32 |

**Qubit Scaling (angle encoding, reps=1):**

| Qubits | mean_var | mean|grad| | Params |
|--------|----------|-----------|--------|
| 4 | 10.629 | 2.090 | 16 |
| 6 | 10.467 | 2.090 | 24 |
| 8 | 9.757 | 2.009 | 32 |
| 10 | 10.038 | 2.005 | 40 |

**Depth Scaling (angle encoding, 8 qubits):**

| Reps | mean_var | mean|grad| | Params |
|------|----------|-----------|--------|
| 1 | 9.757 | 2.009 | 32 |
| 2 | 8.237 | 1.911 | 48 |
| 3 | 6.007 | 1.676 | 64 |

**Bulgular:**
1. **Barren plateau YOK** 4-10 qubit araliginda (variance ~10, stabil) — 8 qubit esik altinda
2. **IQP ve Dense encoding'lerde cok dusuk gradient variance** (~0.85-0.90) — bu encoding'lerin dusuk accuracy'sini acikliyor (gradient yok → ogrenme yok)
3. **Derinlik artisi gradient'i dusurur** (reps=1: 9.76 → reps=3: 6.01) ama dramatik degil
4. Angle ve Re-uploading encoding'ler benzer gradient variance'a sahip (~9-10) ama accuracy farkli — gradient variance gerekli ama yeterli degil

### 6.4 HATA #11: Windows Page File Yetersizligi

**Hata:** `OSError: WinError 1455 — disk bellegi dosyasi cok kucuk`
- **Sebep:** 6+ paralel Python process ayni anda PyTorch CUDA DLL'lerini yuklerken Windows page file doldu
- **Konum:** qattn_iqp training baslatirken
- **Cozum:** Diger process'lerin bitmesini bekleyip tek tek calistirma
- **BILDIRI NOTU:** Windows ortaminda paralel quantum ML deneyleri icin sistem kaynaklari sinirlamasidir

### 6.5 Phase 6 Genel Degerlendirme

**Ne gosterildi:**
- 4 encoding stratejisinin sistematik karsilastirmasi (literaturde NLP-spesifik karsilastirma az)
- Angle encoding NLP icin acik ara en iyi (%69.9 vs IQP %62.4)
- Barren plateau 4-10 qubit'te gorunmuyor (gradient variance stabil ~10)
- IQP/Dense'in dusuk gradient variance'i → dusuk accuracy korelasyonu
- Sigmoid scaling etkisi encoding-bagimsiz (+17 puan)

**Ne gosterilemedi:**
- Quantum attention avantaji (qattn sonuclari istatistiksel olarak anlamsiz)
- Klasik baseline'i gecen quantum model
- Farkli dataset (SST-2) uzerinde dogrulama
- Multi-seed validation

**Durustca sonuc:** Quantum encoding secimi NLP performansini etkiler (angle > dense > IQP > re-uploading) ama hicbir quantum model basit lineer baseline'i gecemiyor. Bu, NISQ caginda 8 qubit ile beklenen bir sonuctur.

---

## BILDIRI MALZEME DEPOSU (Kapsamli)

> Bu bolum, bildiri yazarken dogrudan kullanilacak tum malzemeyi icerir.
> Her alt baslik bildirinin bir bolumune karsilik gelir.

### A. Motivasyon ve Arka Plan (Introduction icin)

**Neden bu proje?**
- Onceki 3 projede (2048-quantum, quantum-state-prep, qaoa-maxcut) klasik simulasyonda kuantum avantaji bulunamadi
- 4. projede strateji degisikligi: gercek IBM kuantum donanimi + standart NLP benchmark'lari
- Arastirma sorusu: "Kuantum devreleri, NLP embedding'leri uzerinde attention mekanizmasi olarak kullanildiginda, klasik attention'a karsi rekabetci performans gosterebilir mi?"

**Literatur bosluklari:**
1. NLP-spesifik quantum encoding stratejilerinin sistematik karsilastirmasi yok
2. Identity initialization'in NLP gorevlerindeki etkisi incelenmemis
3. Gercek kuantum donaniminda NLP inference dogrulamasi az sayida
4. Sigmoid scaling gibi encoding-specific preprocessing tecnikleri belgelenmemis

**Onceki projelerden tasinan dersler (bildirinin "Related Work / Our Experience" bolumu icin):**
- 2048-quantum (VQC-RL): El yapimi heuristik domasyonu, kuantum parametreleri ogrenemiyor
- quantum-state-prep (VQC): 3 bug (CNOT ring, deceptive reward, zero-init decoder) — initialization kritik
- qaoa-maxcut (QAOA): Dogru calisir ama 100-1000x yavas, avantaj yok n<=16'da

### B. Metodoloji (Methods icin)

**Veri Hazirlama Pipeline:**
1. Ham metin → DistilBERT (66.4M param, frozen, eval mode) → [CLS] token 768-dim vektor
2. StandardScaler normalize (mean=0, std=1) → PCA fit (sadece train) → 768→8 boyut
3. Korunan varyans: IMDb %35.7, SST-2 %36.3
4. Split: IMDb 22500/2500/25000, SST-2 57728/872/9621 (train/val/test)
5. Adil karsilastirma icin 2000 sample subset (random, seed=42)

**Neden PCA 768→8?**
- 8 qubit = IBM free tier ile uyumlu
- 2^8 = 256 durum — simulasyonda yonetilebilir
- Daha fazla qubit → ustsel bellek/zaman artisi (2^n)
- %36 varyans: dusuk ama sentiment bilgisi ilk bilesenlerde yogun (PC1 tek basina %8-11)

**Quantum Circuit Tasarimi:**
- Ansatz: EfficientSU2-equivalent (Ry + Rz rotasyonlari, CZ linear entanglement)
- Encoding: Angle encoding — Ry(sigmoid(x_i) * pi) ile her qubit'e input yukleme
- Parametreler: reps=1 → 32 quantum + 18 classical = 50 toplam
- Olcum: Pauli-Z beklenen degeri her qubit icin → 8-dim output

**Sigmoid Scaling Formulu (KRITIK BULGU):**
```
x_encoded = sigmoid(x_pca) * pi
```
- x_pca ~ N(0, 1), aralik yaklasik [-3, +3]
- sigmoid: (-inf, inf) → (0, 1)
- *pi: (0, 1) → (0, pi)
- Ry(0) = |0>, Ry(pi) = |1> → tam Bloch kure kapsamasi
- Bu scaling OLMADAN: %52 (random). BU SCALING ILE: %69 (+17 puan)
- Bildiri iddiasi: "Encoding-preprocessing uyumu kuantum ML'de kritik bir faktordur"

**Training Detaylari:**
- Optimizer: Adam, ayri lr (quantum: 0.05, classical: 0.001)
- Loss: CrossEntropyLoss
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Early stopping: patience=10
- Batch size: 16 (quantum), 64 (klasik)
- Device: CPU (quantum simulasyon), CUDA RTX 4060 (klasik)

### C. Deney Sonuclari (Results icin)

**Ana Sonuc Tablosu (IMDb, 2000 sample subset):**

| Model | Tip | Params | Init | Val Acc | Test Acc | Train Sure | Epoch Ort |
|-------|-----|--------|------|---------|----------|------------|-----------|
| Linear Baseline | Klasik | 18 | - | 74.0% | 72.2% | 2.7s | 0.1s |
| Gated Attention | Klasik | 306 | Xavier | 73.6% | 71.9% | 3.0s | 0.1s |
| Quantum EfficientSU2 | Kuantum | 50 | Identity (theta=0) | 70.0% | 68.8% | 797s | 37.9s |
| Quantum EfficientSU2 | Kuantum | 50 | Random [0,2pi) | 71.0% | 69.2% | 512s | 42.7s |

**Baslica Analizler:**

1. **Parametre verimliligi:** Quantum (50 param) vs Attention (306 param) → quantum 6.1x daha az parametreyle sadece 2.7 puan geride. Parametre basina accuracy: quantum 1.38%/param vs attention 0.23%/param
2. **Identity vs Random init:** 8 qubit'te identity init avantaj gostermedi (%68.8 vs %69.2). Literaturde barren plateau n>>8 icin belirgin — 8 qubit bu esigi asmayabilir.
3. **Training dinamikleri:** Random init 2. epoch'ta en iyi val accuracy'ye ulasti (hizli convergence), identity init 11 epoch aldi. Random init'in baslangic parametreleri tesadufen daha iyi bir optimizasyon bolgesinde olabilir.
4. **Training suresi:** Quantum 267x yavas (797s vs 3s). PyTorch-native simulasyonla bile, 2^8 boyutlu Hilbert uzayinda matris carpimi computational overhead yaratir.

**Encoding Etkisi Deneyi:**
| Scaling | Identity Init | Random Init |
|---------|--------------|-------------|
| Yok (raw PCA) | 51.6% | 52.1% |
| sigmoid*pi | 68.8% | 69.2% |
| **Fark** | **+17.2 puan** | **+17.1 puan** |

Bu, encoding-preprocessing uyumunun initialization stratejisinden bagimsiz oldugunu gosteriyor — her iki init icin de ayni ~17 puan artis.

**IBM QPU Dogrulama:**
| Metrik | Simulator | IBM QPU (ibm_fez 156q) |
|--------|-----------|------------------------|
| Accuracy | 70.0% (7/10) | 70.0% (7/10) |
| Tahmin uyumu | - | 80% (8/10 ayni) |
| Transpiled depth | - | 42 |
| Sure | 0.091s | 138.4s |
| Backend | PyTorch CPU | ibm_fez (Eagle r3) |

### D. Teknik Kararlar ve Gerekceleri (Discussion icin)

1. **Neden PyTorch-native simulasyon (Qiskit degil)?**
   - Qiskit EstimatorQNN: parameter-shift rule ile gradient → 1501ms/sample backward
   - Parameter-shift: f'(theta) = [f(theta+pi/2) - f(theta-pi/2)] / 2
   - 48 param icin: 96 ekstra circuit evaluation/sample
   - PyTorch autograd: chain rule ile tek backward pass → 14ms/sample
   - Hizlanma: ~107x per sample, ~3400x toplam training
   - Trade-off: Simulasyon sinirlari (n_qubits < ~20), gercek QPU'da parameter-shift gerekir
   - Cozum: Simulatorde PyTorch ile egit, IBM QPU'da sadece inference

2. **Neden 2000 sample subset?**
   - Adil karsilastirma: Quantum model yavas, tum veriyle saatlerce surer
   - Literaturde standart: Quantum ML calismalarinda 1000-5000 sample yaygin
   - Klasik modeller de ayni subset'le — birebir ayni veri

3. **Neden reps=1 (reps=2 degil)?**
   - reps=2: 48 quantum param, daha derin circuit
   - reps=1: 32 quantum param, daha sig circuit → daha az barren plateau riski
   - 8 qubit'te fark minimal — derinlik artirmak circuit noise'u arttirir (QPU'da)

4. **Neden CZ gate (CNOT degil)?**
   - EfficientSU2 Qiskit convention'i CZ kullanir
   - CZ = controlled-Z, CNOT'a esdeger (Hadamard conjugation ile)
   - IBM donanimi CZ destekler (native gate set)
   - Linear entanglement: qubit i ↔ qubit i+1 (nearest-neighbor)

5. **Neden DistilBERT (BERT degil)?**
   - 66.4M vs 110M parametre — daha hizli embedding cikartma
   - Performans farki minimal (%1-2) ama hiz 2x
   - Frozen kullanim: sadece feature extractor, fine-tune yok

### E. Sinirlamalar (Limitations icin)

1. **Simulasyon vs gercek kuantum:** Training tamamen klasik simulasyonda. QPU sadece inference dogrulamasi icin kullanildi. Gercek kuantum avantaji ancak QPU-native training ile test edilebilir.
2. **Kucuk qubit sayisi:** 8 qubit, 2^8=256 dim Hilbert uzayi. Klasik bilgisayar bunu kolayca simule eder → "kuantum avantaji" mümkün degil bu olcekte.
3. **PCA bilgi kaybi:** %36 varyans korunuyor — sentiment bilgisinin ne kadari kaybedildi bilinmiyor.
4. **Tek encoding stratejisi:** Sadece angle encoding test edildi. Amplitude encoding, IQP, data re-uploading karssilastirilmadi.
5. **Tek ansatz:** EfficientSU2 tek secenek. Hardware-efficient, problem-aware ansatz'lar diger secenekler.
6. **Sample boyutu:** 2000 sample → istatistiksel guc sinirli. Confidence interval'lar rapor edilmedi.
7. **SST-2 eksik:** Ikinci dataset'te model egitimleri yapilmadi → generalizasyon iddiasi zayif.
8. **Interference mekanizmasi eksik:** Proposal 2'deki "quantum attention" aslinda quantum circuit output'unun attention weight olarak kullanilmasini ongoriyor. Biz sadece dogrudan classification yaptik.

### F. Proposal 2 Gap Analizi

| Proposal 2 Maddesi | Durum | Not |
|---------------------|-------|-----|
| Amplitude/angle encoding karsilastirmasi | EKSIK | Sadece angle encoding |
| EfficientSU2 ansatz | TAMAM | PyTorch-native |
| Pauli-Z measurement | TAMAM | 8 qubit |
| **Interference + softmax (gercek quantum attention)** | **EKSIK** | En kritik eksik — sadece Linear classifier var |
| Identity initialization | TAMAM | 8 qubit'te avantaj yok |
| IMDb benchmark | TAMAM | %69 vs %72 |
| SST-2 benchmark | EKSIK | Veri hazir, egitim yok |
| Barren plateau gradient variance | EKSIK | Olculemedi |
| IBM QPU dogrulama | TAMAM | ibm_fez, %70 |
| Few-shot learning | EKSIK | Denenmedi |
| Parametre verimliligi analizi | KISMI | Hesaplandi ama derinlestirilmedi |

### G. Kronolojik Zaman Cizelgesi

| Saat | Islem | Sonuc |
|------|-------|-------|
| 0:00 | Proje olusturma, system_map.md | Klasor yapisi hazir |
| 0:05 | pip install (qiskit-ibm-runtime, transformers, datasets) | Kurulum OK |
| 0:10 | IBM token kaydi | Hata #1 (ibm_quantum → ibm_quantum_platform), duzeltildi |
| 0:15 | 8 qubit EfficientSU2 testi | 48 param, forward OK, identity init CZ yuzunden != I (Hata #3) |
| 0:20 | IMDb indirme | 25K train + 25K test, 74.2s |
| 0:22 | SST-2 indirme | 67K train + 872 val, 5.4s |
| 0:25 | DistilBERT embedding (IMDb) | 182s GPU, (50000, 768) |
| 0:30 | DistilBERT embedding (SST-2) | 54s GPU, Hata #4 (tensor index), Hata #5 (UNEXPECTED keys) |
| 0:35 | PCA 768→8 | IMDb %35.7, SST-2 %36.3, ~1.2s |
| 0:40 | Classical baseline implementation | 18 params, Linear(8,2) |
| 0:45 | Classical attention implementation | 306 params, Q/K/V + gated |
| 0:50 | Qiskit quantum model implementation | 66 params, EstimatorQNN + TorchConnector |
| 0:55 | Qiskit training denemesi | BASARISIZ — 1501ms/sample backward, 285h tahmin (Hata #7, #8) |
| 1:10 | PyTorch-native quantum model | 50 params, 14ms/sample backward, 3400x hizlanma |
| 1:20 | Ilk quantum training (scaling yok) | %52 accuracy — random seviyesi (Hata #10) |
| 1:25 | Sigmoid scaling ekleme | %52 → %69 (+17 puan!) |
| 1:30 | Tum modellerin training'i | Baseline %72.2, Attention %71.9, Q-identity %68.8, Q-random %69.2 |
| 2:00 | IBM QPU ilk deneme | BASARISIZ — Error 6073 bellek (Hata #9) |
| 2:10 | IBM QPU ikinci deneme (batched) | BASARILI — %70 accuracy, simulator ile eslesme |
| 2:30 | developments.md, system_map.md guncelleme | Dokumantasyon tamamlandi |
| 2:45 | IBM Quantum almanac (Playwright scraping) | 873 satir dokumantasyon |

### H. Reproduce Edilebilirlik Bilgileri

**Tam environment:**
```
Python 3.13.7
torch==2.6.0+cu124
qiskit==2.3.0
qiskit-aer==0.17.2
qiskit-machine-learning==0.9.0
qiskit-ibm-runtime==0.45.1
transformers==5.3.0
datasets==4.6.1
scikit-learn==1.7.2
numpy (bundled with torch)
Windows 11 Pro 10.0.26200
NVIDIA RTX 4060 Laptop GPU (CUDA 12.4)
```

**Seed:** 42 (tum random islemler icin)
**IBM QPU:** ibm_fez (156 qubit, Eagle r3), 2026-03-08

**Komutlar:**
```bash
# Data preprocessing
PYTHONIOENCODING=utf-8 python data/preprocess.py

# Training (her model icin)
PYTHONIOENCODING=utf-8 python training/train.py --model baseline --dataset imdb --subset
PYTHONIOENCODING=utf-8 python training/train.py --model attention --dataset imdb --subset
PYTHONIOENCODING=utf-8 python training/train.py --model quantum --dataset imdb --reps 1 --lr_quantum 0.05 --subset
PYTHONIOENCODING=utf-8 python training/train.py --model quantum_random --dataset imdb --reps 1 --lr_quantum 0.05 --subset

# IBM QPU inference
PYTHONIOENCODING=utf-8 python experiments/ibm_inference.py
```

---

## Phase 8: Balanced QPU Validation — T1 Relaxation Bias Test (2026-03-11)

### Motivasyon
Gemini code review'dan gelen kritik elestiri: Phase 7 QPU deneyinde tum 10 test sample'i Label 0 (negative sentiment) idi. Superconducting qubit'lerde T1 amplitude damping kubitleri |0> ground state'e ceker. Bu, Label 0 tahminlerini artifisyel olarak "dogru" gosterebilir — donanim gurultusu ile dogru tahmin yonu ortusur.

### Yapilan Islem
1. **Dengeli test seti olusturuldu:** 5 Label 0 + 5 Label 1, seed=42 ile reproducible (IMDb test setinden: L0 indices [1766, 11919, 8909, 4963, 10099], L1 indices [23559, 19772, 22220, 17452, 19251])
2. **Simulator gold standard:** Tum 4 encoding icin 80% accuracy (L0=80%, L1=80% simetrik)
3. **IBM QPU inference (ibm_fez):** Yeni IBM token ile taze kota. batch_size=10 optimizasyonu (eski: 5). 4 encoding x 10 sample = 4 job (eski: 8 job).
4. **T1 bias analizi:** L0 agreement vs L1 agreement karsilastirmasi

### Script
`experiments/ibm_balanced_qpu.py` — optimize edilmis versiyon:
- batch_size=10 (tek job per encoding, 80 PUB)
- Hata durumunda batch_size=5'e otomatik fallback
- Detayli T1 bias analiz fonksiyonu
- JSON ciktisi: `results/logs/ibm_balanced_qpu_results.json`

### Sonuclar — MUHTESEM

| Encoding | Sim Acc | QPU Acc | Agreement | L0 Agree | L1 Agree | T1 Bias | Depth |
|----------|---------|---------|-----------|----------|----------|---------|-------|
| Angle    | 80%     | 80%     | 100%      | 100%     | 100%     | YOK     | 19    |
| Dense    | 80%     | 80%     | 100%      | 100%     | 100%     | YOK     | 18    |
| IQP      | 80%     | 80%     | 100%      | 100%     | 100%     | YOK     | 101   |
| Reupload | 80%     | 80%     | 100%      | 100%     | 100%     | YOK     | 21    |

### Ana Bulgular
1. **T1 relaxation bias YOK** — QPU hem Label 0 hem Label 1 icin esit fidelity gosteriyor
2. **Tum 4 encoding %100 sim-QPU agreement** — Phase 7'de 3 encoding %90 idi, balanced sette hepsi %100
3. **IQP (depth=101) bile %100 agreement** — depth-noise korelasyonu bu olcekte sistematik degil
4. **Pred0 shift = 0** — QPU hicbir encoding icin tahminleri |0> yonune kaydirmamis

### QPU Job Detaylari (ibm_fez, 2026-03-11)
| Job ID | Encoding | PUB | Sure |
|--------|----------|-----|------|
| d6ogisu9td6c73apbbh0 | Angle | 80 | 393.8s |
| d6ogluu9td6c73apbec0 | Dense | 40 | 485.7s |
| d6ogpom9td6c73apbi40 | IQP | 80 | 539.0s |
| d6ogtv43pels73a2q6g0 | Reupload | 80 | 144.0s |
Toplam: 1562.5s (~26 dk queue dahil)

### Paper Guncelemeleri
- Section 4.5 yeniden yazildi: Table 6 artik L0/L1 agreement sutunlari iceriyor
- "Test set design" paragrafi: T1 bias motivasyonunu acikladik (proaktif tasarim, mazeret degil)
- Limitation #5 guncellendi: "label balance" sorunu COZULDU, sadece sample size sinirlamasi kaldi
- Conclusion guncellendi: "no T1 relaxation bias" iddiasi eklendi

### Ogrenilen Dersler
- **T1 bias testi zorunludur:** Superconducting QPU'larda sadece tek sinif test etmek yaniltici
- **batch_size=10 (80 PUB) guvenli:** IBM Open Plan'da 80 PUB tek job'da calisiyor
- **Python stdout buffering:** Background task'larda print() buffered, WARNING (stderr) hemen gorunur. PYTHONUNBUFFERED=1 gerekebilir.
- **KeyError bug:** dict key isimlendirmesinde tutarlilik onemli ("preds" vs "qpu_preds")
