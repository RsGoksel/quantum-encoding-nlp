<!--
  SYSTEM_MAP.md — LLM Context File
  ==================================
  Bu dosya projenin tam haritasidir. Her yeni LLM oturumu bu dosyayla baslamalidir.
  Proje yapisi, teknik kararlar, mimari ve ilerleme durumu burada tutulur.

  Proje: Quantum-Enhanced Attention for RAG Pipelines
  Konum: C:\Users\gokss\OneDrive\Masaüstü\Kuantum\quantum-attention\
  Olusturulma: 2026-03-08
  Son guncelleme: 2026-03-08

  Iliskili projeler:
  - ../2048-quantum/         → VQC-RL 2048 (kuantum avantaji yok)
  - ../quantum-state-prep/   → VQC-RL kuantum durum hazirlama (kuantum avantaji yok)
  - ../qaoa-maxcut/           → QAOA vs klasik MaxCut (kuantum avantaji yok)
  - ../tree_project_report.md → 3 projenin kapsamli raporu
  - ../journey-lessons-learned.md → Tum surecin dokumantasyonu
-->

# System Map — Quantum-Enhanced Attention for RAG Pipelines

---

## 1. Project Overview

**Amac:** Identity-initialized kuantum attention mekanizmasi ile RAG (Retrieval-Augmented Generation) pipeline'larinda query-document benzerligini hesaplamak. Klasik self-attention'a karsi kuantum kernelin avantaj saglayip saglamadigini adil ve olculebilir sekilde test etmek.

**Motivasyon:**
- Onceki 3 projede (2048-quantum, quantum-state-prep, qaoa-maxcut) simulasyon olceginde kuantum avantaji bulunamadi
- Bu projede farkli strateji: gercek IBM kuantum donanimi (100+ qubit) uzerinde calisma imkani
- Identity initialization ile barren plateau problemini cozme (Bug 3'ten ogrendi̇gi̇mi̇z ders)
- 8 qubit yeterli — IBM free tier (10 dk QPU / 28 gun) ile uyumlu
- Standart NLP benchmark'lari (IMDb, SST-2) ile olculebilir sonuclar

**Kaynak:** "Quantum Frontiers in the NISQ Era" arastirma briefing'i (Proposal 2)

**Hedef:** Konferans bildirisi (paper) yayinlamak.

---

## 2. Technical Architecture

### 2.1 Pipeline Overview

```
Input Text
    |
    v
[Classical Embedding] — Pre-trained (BERT/DistilBERT frozen)
    |
    v
[Dimensionality Reduction] — PCA: 768 → 8 boyut
    |
    v
[Quantum Encoding] — Angle Encoding veya ZZFeatureMap (8 qubit)
    |
    v
[Quantum Attention Circuit] — EfficientSU2 ansatz, identity-initialized
    |
    v
[Measurement] — Pauli-Z expectation values (8 output)
    |
    v
[Interference + Softmax] — Quantum kernel similarity scores
    |
    v
[Classical Head] — Linear classifier (sentiment/retrieval task)
    |
    v
Output (classification / ranking)
```

### 2.2 Quantum Circuit Design

**Ansatz:** EfficientSU2 (Qiskit built-in)
- Ry, Rz rotation gates per qubit
- CZ entangling gates (linear connectivity)
- Derinlik: 2-4 tekrar katmani (tunable)

**Encoding:**
- **Angle Encoding:** Her qubit'e Ry(x_i) ile veri yukleme — basit, 8 feature = 8 qubit
- **ZZFeatureMap:** Qubit cifti arasinda entanglement-based encoding — daha ifade gucu yuksek

**Identity Initialization (KRITIK):**
- Tum rotation acilari 0 veya 2*pi olarak baslatilir
- Circuit baslangicta identity (I) matrisi gibi davranir
- Gradyanlar sifirdan baslamaz, kuantum parametreleri "klasik noktadan" ogrenmeye baslar
- Bu, quantum-state-prep projesindeki Bug 3'un (zero-init decoder) cozumunun kuantum tarafindaki karsiligi

**Measurement:**
- Her qubit icin Pauli-Z beklenen degeri: <Z_i>
- 8 qubit → 8 boyutlu output vektoru
- Bu vektor attention score olarak kullanilir

### 2.3 Hybrid Training

**Framework:** Qiskit + PyTorch (TorchConnector)
- Qiskit `EstimatorQNN` ile kuantum circuit'i tanimla
- `TorchConnector` ile PyTorch Module'e sar
- Standard backpropagation ile end-to-end training
- Parameter-shift rule ile kuantum gradyanlari hesapla

**Optimizer:** Adam (lr=0.001 klasik, lr=0.01 kuantum — ayri lr'ler)

**Loss:** CrossEntropyLoss (classification task)

---

## 3. Experimental Design

### 3.1 Benchmarks

| Dataset | Task | Metrik | Boyut |
|---------|------|--------|-------|
| IMDb | Sentiment (binary) | Accuracy, F1 | 50K review |
| SST-2 | Sentiment (binary) | Accuracy | 67K sentence |

### 3.2 Models to Compare

| Model | Aciklama | Parametre |
|-------|----------|-----------|
| Classical Baseline | DistilBERT + Linear head | ~66M (frozen) + small head |
| Classical Attention | DistilBERT + learned attention + Linear | ~66M + attention params |
| **Quantum Attention** | DistilBERT + PCA→QNN attention + Linear | ~66M + ~50-100 quantum params |
| Quantum (no identity init) | Ayni ama random initialization | Kontrol grubu |

### 3.3 Metrics

- **Accuracy / F1**: Performans karsilastirmasi
- **Training convergence**: Epoch vs loss/accuracy curves
- **Barren plateau analysis**: Var[dC/dtheta] vs qubit sayisi
- **Identity init vs random init**: Ayni mimaride initialization etkisi
- **QPU time**: IBM quantum uzerinde inference suresi

### 3.4 Hypotheses

1. Identity-init quantum attention, random-init'ten anlamli sekilde daha iyi converge eder
2. Quantum attention, 8-dim PCA uzayinda klasik attention ile yarisabilir performans gosterir
3. Barren plateau etkisi identity-init ile azalir (gradient variance olcumu ile dogrulanir)

---

## 4. Technology Stack

| Katman | Teknoloji | Kurulu Versiyon |
|--------|-----------|-----------------|
| Quantum SDK | Qiskit | 2.3.0 |
| Quantum ML | qiskit-machine-learning | 0.9.0 |
| IBM Runtime | qiskit-ibm-runtime | 0.45.1 |
| Deep Learning | PyTorch | 2.6.0+cu124 (CUDA) |
| NLP | HuggingFace Transformers | 5.3.0 |
| Datasets | HuggingFace datasets | 4.6.1 |
| Dim. Reduction | scikit-learn (PCA) | 1.7.2 |
| Quantum Hardware | IBM Quantum (free tier) | Eagle r3 (133-156 qubit) |
| Visualization | matplotlib 3.10.7, seaborn 0.13.2 | latest |
| Experiment Tracking | JSON logs (onceki projelerle ayni) | — |
| Python | CPython | 3.13.7 |
| OS | Windows 11 Pro | 10.0.26200 |

---

## 5. Project Structure (Planned)

```
quantum-attention/
├── system_map.md              ← Bu dosya (LLM context)
├── updates.md                 ← Ilerleme kaydi (kronolojik)
├── requirements.txt           ← Python dependencies
│
├── data/
│   ├── download.py            ← Dataset indirme (HuggingFace)
│   └── preprocess.py          ← Embedding extraction + PCA
│
├── models/
│   ├── classical_baseline.py  ← DistilBERT + Linear
│   ├── classical_attention.py ← DistilBERT + learned attention
│   ├── quantum_attention.py   ← QNN attention head (EfficientSU2)
│   └── hybrid_model.py        ← Full hybrid pipeline
│
├── quantum/
│   ├── circuit.py             ← Quantum circuit tanimlari
│   ├── encoding.py            ← Angle/ZZFeatureMap encoding
│   └── identity_init.py       ← Identity initialization logic
│
├── training/
│   ├── train.py               ← Training loop
│   ├── evaluate.py            ← Evaluation + metrics
│   └── config.py              ← Hyperparameters
│
├── experiments/
│   ├── run_comparison.py      ← Tum modelleri calistir + karsilastir
│   ├── barren_plateau.py      ← Gradient variance analizi
│   └── ibm_balanced_qpu.py   ← Phase 8: Balanced QPU inference (T1 bias test), 5 Label 0 + 5 Label 1
│
├── results/
│   ├── logs/
│   │   ├── ...                ← JSON experiment logs
│   │   └── ibm_balanced_qpu_results.json ← Phase 8 QPU results: 100% agreement all encodings
│   └── plots/                 ← Visualization PNG'leri
│       ├── ...
│       ├── FIGURE-Pipeline diagram.jpg          ← Pipeline diagram (DistilBERT → PCA → Quantum → Classifier)
│       ├── FIGURE_4_circuit_diagrams.jpg        ← 4 encoding circuit diagrams + IBM transpilation view
│       └── IBM_circuit_screenshot.jpg           ← IBM Quantum dashboard: Angle encoding transpiled on ibm_fez
│
└── checkpoints/               ← Model checkpoints
```

---

## 6. Key Risks and Mitigations

| Risk | Etki | Mitigasyon |
|------|------|------------|
| Barren plateau (random init) | Training basarisizligi | Identity initialization (ana hipotez) |
| PCA bilgi kaybi (768→8) | Dusuk performans | PCA variance ratio kontrolu, farkli boyutlar dene |
| IBM QPU kuyruk suresi | Yavas iterasyon | Simulatorde gelistir, sadece final deneyleri QPU'da |
| 8 qubit yetersiz ifade gucu | Klasikten kotu sonuc | EfficientSU2 derinlik artirma, encoding degistirme |
| Qiskit API degisiklikleri | Kod kirilmasi | Versiyon pinleme, documentation takibi |
| Overfitting (az kuantum param) | Yaniltici sonuc | Cross-validation, train/val/test ayirimi |

---

## 7. Lessons from Previous Projects (KRITIK)

Bu projeye onceki 3 projeden tasidiklarimiz:

1. **Unitarity dogrulama** (qaoa-maxcut): Her quantum circuit'in U†U = I oldugunu test et
2. **Deceptive reward/metric** (quantum-state-prep): Tek bir metrige guvenme, birden fazla metrik kullan
3. **Zero-init / identity-init** (quantum-state-prep Bug 3): Initialization stratejisi kritik — bu projenin ana hipotezi
4. **Adil karsilastirma** (tum projeler): Klasik baseline ayni kosullarda, ayni veri ile
5. **Heuristic-free** (2048-quantum): El yapimi heuristic'ler olmadan saf ogrenmeli yaklasim
6. **Reproducibility** (tum projeler): Fixed seed, JSON logs, checkpoint kaydi
7. **COBYLA outlier** (qaoa-maxcut): Optimizer stuck olabilir — timeout veya alternative optimizer hazirla
8. **Windows encoding** (tum projeler): PYTHONIOENCODING=utf-8, unicode karakter kullanma

---

## 8. Implementation Phases

### Phase 1: Altyapi [TAMAMLANDI]
- [x] Proje klasoru olusturma
- [x] system_map.md olusturma
- [x] Qiskit 2.3 + qiskit-machine-learning 0.9 + qiskit-ibm-runtime 0.45 kurulumu
- [x] Transformers 5.3 + datasets 4.6 kurulumu
- [x] IBM Quantum hesap dogrulama (3 QPU mevcut, token kaydedildi)
- [x] 8 qubit EfficientSU2 circuit testi (48 trainable param, forward pass OK)
- [x] TorchConnector entegrasyonu testi (PyTorch backward pass OK)

### Phase 2: Data Pipeline [TAMAMLANDI]
- [x] IMDb indirme: 50K review, 74.2s
- [x] SST-2 indirme: 67K cumle, 5.4s
- [x] DistilBERT embedding (GPU RTX 4060): IMDb 182s, SST-2 54s
- [x] PCA 768->8: IMDb %35.7 varyans, SST-2 %36.3 varyans, ~1.2s
- [x] Train/val/test split (seed=42)
- [x] Kayit: imdb_embeddings.pt (1956 KB), sst2_embeddings.pt (2667 KB)
- **Hata:** HuggingFace Dataset tensor index kabul etmiyor -> .tolist() ile cozuldu
- **Not:** PCA %36 varyans dusuk gorunebilir ama 8 qubit siniri var

### Phase 3: Model Implementation [TAMAMLANDI]
- [x] Classical baseline (Linear 8->2, 18 params)
- [x] Classical attention model (Q/K/V projection, 306 params)
- [x] Quantum circuit: PyTorch-native EfficientSU2-equivalent (quantum_attention_fast.py)
- [x] Identity init + random init varyantlari
- [x] Sigmoid scaling: PCA output -> [0,pi] (angle encoding uyumu)
- **Kritik karar:** Qiskit EstimatorQNN yerine PyTorch-native simulasyon (3400x hizlanma)

### Phase 4: Training & Experiments [TAMAMLANDI]
- [x] Classical baseline training: 2000 subset, 30 epoch -> Val %74.0, Test %72.2 (2.7s)
- [x] Classical attention training: 2000 subset, 28 epoch -> Val %73.6, Test %71.9 (3.0s)
- [x] Quantum identity init: 2000 subset, 21 epoch -> Val %70.0, Test %68.8 (797s)
- [x] Quantum random init: 2000 subset, 12 epoch -> Val %71.0, Test %69.2 (512s)
- **Sonuc:** Quantum model %69, klasik %72 — yakin ama geride
- **Surpriz:** Random init identity init'ten biraz daha iyi (beklentinin tersi)
- **BILDIRI NOTU - Sigmoid Scaling Etkisi:** Encoding uyumu olmadan quantum %52 (random), sigmoid scaling ile %69 (+17 puan). Kuantum ML'de veri on-isleme kritik.
- **BILDIRI NOTU - Identity vs Random Init:** 8 qubit'te identity init beklenen avantaji gostermedi (%68.8 vs %69.2). Literaturde barren plateau n>>8 icin belirgin — 8 qubit bu esigi asmayabilir.
- **BILDIRI NOTU - Parametre Verimliligi:** 50 parametreli quantum model, 306 parametreli klasik attention'a ~3 puan farkla yaklasti. Parametre basina performans quantum lehine.
- **BILDIRI NOTU - Training Hizi:** PyTorch-native simulasyon Qiskit parameter-shift'ten 3400x hizli. Quantum ML arastirmalarinda framework secimi kritik.
- [ ] Barren plateau gradient variance analizi (PLANLANACAK)
- [x] IBM QPU inference — Ilk deneme Error 6073 (bellek), duzeltildi, 2. deneme yapilacak

### Phase 4.5 Degerlendirme
- [x] IBM QPU inference basarili (ibm_fez, %70 accuracy, simulator ile eslesme)
- **KRITIK TESPIT:** Phase 1-4.5 = verification (dogrulama) calismasi, literatur katkisi degil.
- Eksik: Sistematik encoding karsilastirmasi, farkli ansatz/qubit sayilari, avantaj gosterimi

### Phase 6: Sistematik Encoding Karsilastirmasi + Quantum Attention [TAMAMLANDI]

**Arastirma Sorusu:** "NLP embedding'leri icin hangi kuantum encoding stratejisi en iyi performansi verir ve kuantum circuit output'u gercek bir attention mekanizmasi olarak nasil kullanilir?"

**6.1 Encoding Sonuclari (IMDb, 2000 subset):**

| Encoding | Qubit | Params | Test Acc | Gradient Var |
|----------|-------|--------|----------|-------------|
| Angle (sigmoid*pi) | 8 | 50 | **69.9%** | 9.757 (yuksek — iyi) |
| Dense Angle | 4 | 26 | 68.1% | 0.855 (dusuk — sorunlu) |
| IQP | 8 | 50 | 62.4% | 0.899 (dusuk — sorunlu) |
| Re-uploading | 8 | 50 | 58.1% | 9.098 (yuksek — iyi) |

**6.2 Quantum Attention Sonuclari:**

| Model | Test Acc | vs Baseline |
|-------|----------|-------------|
| qattn_angle | 73.4% | +1.2% |
| qattn_iqp | 72.8% | +0.6% |
| qattn_reupload | 73.2% | +1.0% |
| Baseline (Linear) | 72.2% | referans |

**DEGERLENDIRME:** qattn sonuclari istatistiksel olarak anlamsiz (tek seed, +1.2% fark, ek klasik parametreler). Kullanici da skeptik.

**6.3 Barren Plateau:** 4-10 qubit'te gradient variance stabil (~10). Barren plateau gorunmuyor.

**6.4 Eksikler:**
- SST-2 deneyleri yapilmadi
- Multi-seed validation yapilmadi
- En iyi encoding icin IBM QPU dogrulamasi yapilmadi

### Phase 8: Balanced QPU Validation [TAMAMLANDI — 2026-03-11]
- [x] T1 bias testi: 5 Label 0 + 5 Label 1 dengeli test seti ile IBM ibm_fez'de inference
- [x] **Sonuc:** T1 bias = YOK. Tum 4 encoding %100 simulator-QPU agreement (balanced set)
- [x] Script: `experiments/ibm_balanced_qpu.py`
- [x] Sonuc: `results/logs/ibm_balanced_qpu_results.json`

### Phase 5: Analysis & Paper (Phase 6 sonrasi)
- [ ] Sonuclari karsilastir ve visualize et
- [ ] Bildiri taslagi yaz
- [ ] Plot'lari paper-quality'ye getir

---

## 9. IBM Quantum Access

- **Plan:** Open (ucretsiz)
- **QPU Suresi:** 10 dakika / 28 gun (sadece islem suresi, kuyruk/baglanti suresi haric)
- **Token:** Kaydedildi (QiskitRuntimeService.save_account ile)
- **Channel:** `ibm_quantum_platform` (eski `ibm_quantum` degil — Qiskit 2.x degisikligi)
- **Mevcut QPU'lar (2026-03-08):**
  - `ibm_fez`: 156 qubit, operational
  - `ibm_marrakesh`: 156 qubit, operational
  - `ibm_torino`: 133 qubit, operational
- **Erisim:** quantum.ibm.com → IBM Quantum Platform
- **Strateji:** Gelistirme tamamen simulatorde (StatevectorEstimator), sadece final validation IBM QPU'da
- **Not:** 8 qubit projemiz icin 133-156 qubit makineler fazlasiyla yeterli

---

## 10. Known Issues & Fixes

### Karsilasilan Hatalar

| # | Hata | Sebep | Cozum |
|---|------|-------|-------|
| 1 | `InvalidAccountError: "Invalid channel value... got 'ibm_quantum'"` | Qiskit 2.x'te channel ismi degisti | `channel='ibm_quantum_platform'` kullan (`ibm_quantum` degil) |
| 2 | `ZZFeatureMap` / `EfficientSU2` DeprecationWarning | Qiskit 2.1+ class-based API deprecated | `from qiskit.circuit.library import zz_feature_map, efficient_su2` fonksiyon API'sine gec |
| 3 | Identity init test: max diff = 1.0 (identity degil) | CZ entangling gate'ler parametresiz, her zaman entanglement yapar | Beklenen davranis — "identity init" sadece trainable rotation parametrelerinin (Ry, Rz) sifir olmasi demek. CZ gate'ler sabit yapi saglar. |
| 4 | `TypeError: len() of a 0-d tensor` SST-2 processing'de | HuggingFace Dataset nesnesi PyTorch tensor index kabul etmiyor | `perm[idx].tolist()` ile int listesine cevir |
| 5 | DistilBERT "UNEXPECTED keys" uyarisi | Masked LM head katmanlari yukleniyor ama DistilBertModel bunlari kullanmiyor | Guvenlice yok sayilabilir — feature extraction etkilenmiyor |
| 6 | `torch.load weights_only=True` numpy hatasi | PCA variance numpy array, PyTorch 2.6 default weights_only=True | `weights_only=False` kullan (kendi dosyamiz, guvenli) |
| 7 | Quantum training 285 saat surer tahmin (22500 sample) | Parameter-shift rule: 48 param x 2 = 96 ek circuit eval/sample, backward 83x yavas | PyTorch-native simulasyona gecildi (quantum_attention_fast.py) — autograd ile ~3400x hizlanma |
| 8 | Qiskit EstimatorQNN backward: 1501ms/sample | Parameter-shift rule: f'(t) = [f(t+pi/2) - f(t-pi/2)] / 2, her param icin 2 ek eval | PyTorch autograd: tek backward pass, 14ms/sample (8 qubit). Training: Qiskit 285h -> PyTorch 5dk |
| 9 | IBM QPU Error 6073: job memory limit exceeded | 50 sample x 8 obs = 400 PUB tek job'da gonderildi, klasik kontrol HW bellek limiti | 5'erli batch (5x8=40 PUB/job) + sample=10'a dusur. **BILDIRI NOTU:** Bu hata NISQ cagi donanim sinirlamalarinin somut bir ornegi — kuantum islemci 156 qubit desteklerken, klasik kontrol donanimi job boyutunu sinirliyor. Gercek kuantum hesaplamanin bottleneck'i her zaman kuantum islemci degil. |
| 10 | Quantum model ilk deney %52 (random seviyesi) | PCA output [-3,+3], angle encoding [0,pi] bekliyor, uyumsuz aralik | sigmoid(x)*pi scaling ile [0,pi]'ye normalize et -> %52->%69. **BILDIRI NOTU:** Veri on-isleme (encoding uyumu) kuantum ML'de kritik — yanlis scaling tum ogrenmeyi yok edebilir. Bu, klasik ML'de normalization'in oneminin kuantum karsiligi. |

### Onemli Notlar
- Windows'ta her Python calistirmada `PYTHONIOENCODING=utf-8` kullan
- Qiskit 2.x function-based API tercih et (class-based deprecated)
- `StatevectorEstimator()` simulatorde kullan, IBM QPU'da `Estimator` (runtime) kullan

---

## 11. Performance Estimates (Phase Sureleri)

| Phase | Islem | Tahmini Sure | Aciklama |
|-------|-------|-------------|----------|
| **Phase 2** | IMDb indirme | ~30-60s | HuggingFace cache'den ~650MB |
| **Phase 2** | SST-2 indirme | ~5-10s | Kucuk dataset (~7MB) |
| **Phase 2** | DistilBERT embedding (50K IMDb) | ~10-20 dk | GPU ile batch processing, frozen model |
| **Phase 2** | DistilBERT embedding (67K SST-2) | ~15-25 dk | Ayni islem, biraz daha fazla veri |
| **Phase 2** | PCA fit + transform | ~1-3s | 768→8, scikit-learn cok hizli |
| **Phase 3** | Classical baseline training (5 epoch) | ~2-5 dk | Linear head, frozen embeddings |
| **Phase 3** | Quantum circuit build | ~100-500ms | 8 qubit, 48 param |
| **Phase 4** | Quantum model training (1 epoch, sim) | ~30-60 dk | Parameter-shift rule yuzunden yavas |
| **Phase 4** | Quantum forward+backward (1 sample, PyTorch-native) | ~14ms | autograd, 8 qubit |
| **Phase 4** | Quantum forward+backward (1 sample, Qiskit param-shift) | ~1520ms | parameter-shift rule, 8 qubit — KULLANILMIYOR |
| **Phase 4** | Quantum forward pass (1 sample, QPU) | ~1-5s | Transpile + queue + execution |
| **Phase 4** | Barren plateau analizi | ~5-10 dk | 100 random init, gradient sampling |
| **Phase 5** | Plot uretimi | ~2-5s | matplotlib, 4-5 figure |

**Toplam tahmini sure:** ~2-4 saat (simulatorde), + ~30 dk IBM QPU deneyleri

---

## 12. Success Criteria

Bildiri icin yeterli sonuclar:

1. **Minimum:** Identity init vs random init arasinda istatistiksel anlamli fark gostermek
2. **Orta:** Quantum attention'in klasik attention ile yarisabilir accuracy vermesi
3. **Ideal:** Quantum attention'in belirli kosullarda (kucuk veri, yuksek boyut) avantaj gostermesi
4. **Her durumda:** Rigorous karsilastirma + analiz + negatif sonuc bile degerli (onceki projeler gibi)

---

## 13. LLM Prompt Tercihleri (Kullanici Talepleri)

- **Dil:** Turkce iletisim, Ingilizce kod
- **Detay seviyesi:** Her ilerleme adiminda detayli aciklama yap — ne yapildi, neden yapildi, ne sonuc verdi
- **Ogretici mod:** Kullanici her seyi ogrenmek istiyor — teknik kararlari, alternatifleri, trade-off'lari acikla
- **Hata raporlama:** Karsilasilan her hatayi, sebebini ve cozumunu yaz (system_map'e de kaydet)
- **Sure bilgisi:** Her islemin ne kadar surdugunu raporla
- **Windows:** PYTHONIOENCODING=utf-8 her zaman kullan
- **IBM Token:** Kaydedildi, her oturum basinda tekrar girmesine gerek yok
