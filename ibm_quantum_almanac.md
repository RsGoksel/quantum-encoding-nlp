# IBM Quantum Almanac

> Bu belge, IBM Quantum Platform dokumantasyonundan derlenmis kapsamli bir referans kilavuzudur.
> Teknik terimler ve kod ornekleri orijinal Ingilizce halleriyle korunmustur.
> Tarih: 2026-03-08

---

## Icindekiler

1. [Bolum 1: Giris (Introduction)](#bolum-1-giris)
   - [Qiskit ve IBM Quantum'a Giris](#qiskit-ve-ibm-quantuma-giris)
   - [Baslangic](#baslangic)
   - [Yetenekleri Kesfetme](#yetenekleri-kesfetme)
   - [Destek](#destek)
2. [Bolum 2: Egitimler (Tutorials)](#bolum-2-egitimler)
   - [Baslangic Egitimi](#baslangic-egitimi)
   - [Avantaja Yonelik Is Akislari](#avantaja-yonelik-is-akislari)
   - [Qiskit Yeteneklerinden Yararlanma](#qiskit-yeteneklerinden-yararlanma)
3. [Bolum 3: Dinamik Devrelerle Bell Cifti Karsilastirmasi](#bolum-3-dinamik-devrelerle-bell-cifti-karsilastirmasi)
   - [Arka Plan](#bell-arka-plan)
   - [Gereksinimler ve Kurulum](#bell-gereksinimler)
   - [Adim 1: Kuantum Problemine Esleme](#bell-adim-1)
   - [Adim 2: Donanim Icin Optimizasyon](#bell-adim-2)
   - [Adim 3: Qiskit Primitives ile Calistirma](#bell-adim-3)
   - [Adim 4: Sonuclarin Islenmesi](#bell-adim-4)
4. [Bolum 4: Hamiltonian Simulasyonu Icin Derleme Yontemleri](#bolum-4-hamiltonian-simulasyonu)
   - [Arka Plan](#ham-arka-plan)
   - [Derleme Yontemlerine Genel Bakis](#ham-yontemler)
   - [Bolum 1: Efficient SU2 Devresi](#ham-su2)
   - [Bolum 2: Hamiltonian Simulasyon Devresi](#ham-hamiltonian)
   - [Ozet ve Oneriler](#ham-ozet)

---

## Bolum 1: Giris

**Orijinal Sayfa:** [https://quantum.cloud.ibm.com/docs/en/guides](https://quantum.cloud.ibm.com/docs/en/guides)

### Qiskit ve IBM Quantum'a Giris

Qiskit, IBM Quantum Platform ve ilgili paketler icin resmi dokumantasyona hos geldiniz. Bu dokumantasyon, araclari kullanmaya baslamaniz icin nasil-yapilir kilavuzlari, uctan uca ornekler iceren kullanim senaryosu egitimlerini ve API referanslarinin bir koleksiyonunu icerir.

**Qiskit**, algoritmalar, yuksek performansli hesaplama ve kuantum bilgi bilimi alanlarinda kuantum arastirma ve gelistirme icin moduler ve genisletilebilir bir cerceve saglar. Arastirmacilar, [ozellestirilmis eklentiler](https://quantum.cloud.ibm.com/docs/guides/addons), yazilim araclari ve kapsamli kaynaklarla kuantum is akislarini olusturabilir, optimize edebilir ve calistirabilir.

**IBM Quantum Platform** uzerinden kullanicilar, Qiskit Runtime ve Qiskit Functions Catalog gibi [kuantum hesaplama hizmetlerine](https://quantum.cloud.ibm.com/docs/guides/compute-services) erisebilir ve is yuklerini IBM kuantum bilgisayar filosunda verimli bir sekilde calistirabilir.

Qiskit ve ilgili paketlerinin otesinde, islevselligini genisletmek icin Qiskit ile arayuz olusturan acik kaynakli projelerin katalogu olan [Qiskit ekosistemi](https://www.ibm.com/quantum/ecosystem) bulunmaktadir.

### Baslangic

| Kaynak | Aciklama |
|--------|----------|
| [Quickstart](https://quantum.cloud.ibm.com/docs/guides/quick-start) | Iki dakikadan kisa surede bir kuantum devresi olusturun - giris veya API anahtari gerekmez. |
| [Tutorials](https://quantum.cloud.ibm.com/docs/tutorials) | Qiskit'i yaygin kuantum hesaplama kullanim senaryolarina uygulayan uctan uca ornekleri deneyin. |

### Yetenekleri Kesfetme

| Yetenek | Aciklama | Baglanti |
|---------|----------|----------|
| **Circuit building** | Qiskit SDK kullanarak kuantum devreleri olusturun. | [Devreleri Olusturma](https://quantum.cloud.ibm.com/docs/guides/construct-circuits) |
| **Optimization** | Qiskit SDK transpiler ile yurutmeye hazir, azaltilmis derinlikte yuksek sadakatli devreler olusturun. | [Transpiler Detaylari](https://quantum.cloud.ibm.com/docs/guides/transpile) |
| **Error mitigation** | IBM Quantum'dan mevcut hata azaltma ve baskilama tekniklerini kesfedin. | [Hata Azaltma](https://quantum.cloud.ibm.com/docs/guides/error-mitigation-and-suppression-techniques) |
| **Execution** | Qiskit Runtime primitives kullanarak kuantum devrelerini IBM Quantum donaniminda calistirin. | [Primitives](https://quantum.cloud.ibm.com/docs/guides/primitives) |
| **Post-processing** | Qiskit eklentileri olarak mevcut olan ozellestirilmis uygulamalar icin son-isleme teknikleriyle sonuclarinizi iyilestirin. | [SQD Addon](https://quantum.cloud.ibm.com/docs/guides/qiskit-addons-sqd) |
| **Qiskit Functions** | Ortak kuruluslar tarafindan olusturulan onceden hazirlanmis araclarla kuantum is yuku olusturmayi hizlandirin. | [Functions](https://quantum.cloud.ibm.com/docs/guides/functions) |

### Dokumantasyon Yan Menusundeki Ana Bolumler

Dokumantasyonun yan menusu su ana bolumleri icerir:

**Baslangic:**
- Introduction
- Quickstart
- IBM Quantum Composer
- Latest updates

**Qiskit:**
- Introduction to Qiskit
- Install
- Circuits and operators
- Transpilation
- Primitives
- Debugging (Yeni)
- Advanced techniques
- Create a provider
- Integrations (Yeni)
- Qiskit code assistant

**IBM Quantum Compute:**
- Introduction to IBM Quantum services
- Run your first circuit on hardware
- IBM Quantum Platform
- Execute with primitives
- IBM quantum computers
- Manage noise
- Manage workload execution
- Execution modes
- Qiskit Serverless
- Qiskit Transpiler Service

**Uygulama Arastirma Araclari:**
- Introduction to Qiskit Functions
- Circuit functions
- Application functions
- Qiskit Function templates
- Sample-based quantum diagonalization (SQD)

**Ek Kaynaklar:**
- Support and FAQ
- Development workflow
- Migration guides
- Open-source resources
- Security and compliance

### Destek

| Kaynak | Aciklama |
|--------|----------|
| [Error code registry](https://quantum.cloud.ibm.com/docs/errors) | Hata kodlarini ve onerilen cozumleri arayin. |
| [Support page](https://quantum.cloud.ibm.com/docs/guides/support) | Sikca sorulan sorularin cevaplarini bulun, hatalari nasil bildirecegi ogren, topluluklara katilin. |

---

## Bolum 2: Egitimler

**Orijinal Sayfa:** [https://quantum.cloud.ibm.com/docs/en/tutorials](https://quantum.cloud.ibm.com/docs/en/tutorials)

Bu egitimler, Qiskit'i yaygin kuantum hesaplama kullanim senaryolarina nasil uygulayacaginizi ogretir.

- **Baslangic** bolumundeki egitimlerle baslayin (kuantum bilgisayarda ilk kez kod calistiriyorsaniz).
- **Avantaja yonelik is akislari** bolumu, gercek dunya problemlerini cozmek icin kuantum bilgisayar kullanan uctan uca ornekler icerir.
- **Qiskit yetenekleri** bolumu, belirli bir is akisinin bir bolumunu veya tamamini iyilestirmek icin Qiskit ekosistemindeki en son ve en gelismis teknikleri kullanan ornekleri icerir.

### Baslangic Egitimi

| Egitim | Baglanti |
|--------|----------|
| CHSH inequality | [Link](https://quantum.cloud.ibm.com/docs/tutorials/chsh-inequality) |

### Avantaja Yonelik Is Akislari

Buyuk olcekli kuantum algoritma gosterimlerini kapsayan egitimler.

#### Dogrulanabilir Ornekleme Algoritmalari (Verifiable Sampling Algorithms)

| Egitim | Baglanti |
|--------|----------|
| Sample-based quantum diagonalization of a chemistry Hamiltonian | [Link](https://quantum.cloud.ibm.com/docs/tutorials/sample-based-quantum-diagonalization) |
| Sample-based Krylov quantum diagonalization of a fermionic lattice model | [Link](https://quantum.cloud.ibm.com/docs/tutorials/sample-based-krylov-quantum-diagonalization) |
| Quantum approximate optimization algorithm | [Link](https://quantum.cloud.ibm.com/docs/tutorials/quantum-approximate-optimization-algorithm) |
| Advanced techniques for QAOA | [Link](https://quantum.cloud.ibm.com/docs/tutorials/advanced-techniques-for-qaoa) |
| Pauli correlation encoding to reduce Maxcut requirements | [Link](https://quantum.cloud.ibm.com/docs/tutorials/pauli-correlation-encoding-for-qaoa) |

#### Gozlenebilir Tahmini (Observable Estimation)

| Egitim | Baglanti |
|--------|----------|
| Krylov quantum diagonalization of lattice Hamiltonians | [Link](https://quantum.cloud.ibm.com/docs/tutorials/krylov-quantum-diagonalization) |
| Nishimori phase transition | [Link](https://quantum.cloud.ibm.com/docs/tutorials/nishimori-phase-transition) |
| Ground-state energy estimation of the Heisenberg chain with VQE | [Link](https://quantum.cloud.ibm.com/docs/tutorials/spin-chain-vqe) |
| Quantum kernel training | [Link](https://quantum.cloud.ibm.com/docs/tutorials/quantum-kernel-training) |
| Enhance feature classification using projected quantum kernels | [Link](https://quantum.cloud.ibm.com/docs/tutorials/projected-quantum-kernels) |

#### Hata Toleransli Algoritmalar (Fault-Tolerant Algorithms)

| Egitim | Baglanti |
|--------|----------|
| Shor's algorithm | [Link](https://quantum.cloud.ibm.com/docs/tutorials/shors-algorithm) |
| Grover's algorithm | [Link](https://quantum.cloud.ibm.com/docs/tutorials/grovers-algorithm) |

### Qiskit Yeteneklerinden Yararlanma

Kuantum algoritmalarini calistirirken performansi, guvenilirligi ve hizi artiran gelismis yetenekleri tanitan bolum.

#### Is Yuku Optimizasyonu (Workload Optimization)

| Egitim | Baglanti |
|--------|----------|
| Benchmark dynamic circuits with cut Bell pairs | [Link](https://quantum.cloud.ibm.com/docs/tutorials/edc-cut-bell-pair-benchmarking) |
| Introduction to fractional gates | [Link](https://quantum.cloud.ibm.com/docs/tutorials/fractional-gates) |
| Qiskit AI-powered transpiler service introduction | [Link](https://quantum.cloud.ibm.com/docs/tutorials/ai-transpiler-introduction) |
| Transpilation optimizations with SABRE | [Link](https://quantum.cloud.ibm.com/docs/tutorials/transpilation-optimizations-with-sabre) |
| Long-range entanglement with dynamic circuits | [Link](https://quantum.cloud.ibm.com/docs/tutorials/long-range-entanglement) |
| Compilation methods for Hamiltonian simulation circuits | [Link](https://quantum.cloud.ibm.com/docs/tutorials/compilation-methods-for-hamiltonian-simulation-circuits) |
| Simulation of kicked Ising Hamiltonian with dynamic circuits | [Link](https://quantum.cloud.ibm.com/docs/tutorials/dc-hex-ising) |

#### Qiskit Functions

| Egitim | Baglanti |
|--------|----------|
| Perform dynamic portfolio optimization with Global Data Quantum's Portfolio Optimizer | [Link](https://quantum.cloud.ibm.com/docs/tutorials/global-data-quantum-optimizer) |
| Error mitigation with the IBM Circuit function | [Link](https://quantum.cloud.ibm.com/docs/tutorials/error-mitigation-with-qiskit-functions) |
| Higher-order binary optimization with Q-CTRL's Optimization Solver | [Link](https://quantum.cloud.ibm.com/docs/tutorials/solve-higher-order-binary-optimization-problems-with-q-ctrls-optimization-solver) |
| Model a flowing non-viscous fluid using QUICK-PDE | [Link](https://quantum.cloud.ibm.com/docs/tutorials/colibritd-pde) |
| Dissociation PES curves with Qunova HiVQE | [Link](https://quantum.cloud.ibm.com/docs/tutorials/qunova-hivqe) |
| Transverse-Field Ising Model with Q-CTRL's Performance Management | [Link](https://quantum.cloud.ibm.com/docs/tutorials/transverse-field-ising-model) |
| Quantum Phase Estimation with Q-CTRL's Qiskit Functions | [Link](https://quantum.cloud.ibm.com/docs/tutorials/quantum-phase-estimation-qctrl) |
| Solve the Market Split problem with Kipu Quantum's Iskay Quantum Optimizer | [Link](https://quantum.cloud.ibm.com/docs/tutorials/solve-market-split-problem-with-iskay-quantum-optimizer) |
| Hybrid quantum-enhanced ensemble classification (grid stability workflow) | [Link](https://quantum.cloud.ibm.com/docs/tutorials/sml-classification) |
| Simulate 2D tilted-field Ising with the QESEM function | [Link](https://quantum.cloud.ibm.com/docs/tutorials/qedma-2d-ising-with-qesem) |
| Simulate a kicked Ising model with the TEM function | [Link](https://quantum.cloud.ibm.com/docs/tutorials/simulate-kicked-ising-tem) |

#### Qiskit Addons

| Egitim | Baglanti |
|--------|----------|
| Multi-product formulas to reduce Trotter error | [Link](https://quantum.cloud.ibm.com/docs/tutorials/multi-product-formula) |
| Approximate quantum compilation for time evolution circuits | [Link](https://quantum.cloud.ibm.com/docs/tutorials/approximate-quantum-compilation-for-time-evolution) |
| Operator backpropagation (OBP) for estimation of expectation values | [Link](https://quantum.cloud.ibm.com/docs/tutorials/operator-back-propagation) |
| Wire cutting for expectation values estimation | [Link](https://quantum.cloud.ibm.com/docs/tutorials/wire-cutting) |
| Circuit cutting for periodic boundary conditions | [Link](https://quantum.cloud.ibm.com/docs/tutorials/periodic-boundary-conditions-with-circuit-cutting) |
| Circuit cutting for depth reduction | [Link](https://quantum.cloud.ibm.com/docs/tutorials/depth-reduction-with-circuit-cutting) |
| Readout error mitigation for the Sampler primitive using M3 | [Link](https://quantum.cloud.ibm.com/docs/tutorials/readout-error-mitigation-sampler) |

#### Hata Azaltma (Error Mitigation)

| Egitim | Baglanti |
|--------|----------|
| Utility-scale error mitigation with probabilistic error amplification | [Link](https://quantum.cloud.ibm.com/docs/tutorials/probabilistic-error-amplification) |
| Combine error mitigation options with the Estimator primitive | [Link](https://quantum.cloud.ibm.com/docs/tutorials/combine-error-mitigation-techniques) |
| Real-time benchmarking for qubit selection | [Link](https://quantum.cloud.ibm.com/docs/tutorials/real-time-benchmarking-for-qubit-selection) |

#### Hata Tespiti (Error Detection) - Yeni

| Egitim | Baglanti |
|--------|----------|
| Repetition codes | [Link](https://quantum.cloud.ibm.com/docs/tutorials/repetition-codes) |
| Low-overhead error detection with spacetime codes | [Link](https://quantum.cloud.ibm.com/docs/tutorials/ghz-spacetime-codes) |

---

## Bolum 3: Dinamik Devrelerle Bell Cifti Karsilastirmasi

**Orijinal Sayfa:** [https://quantum.cloud.ibm.com/docs/en/tutorials/edc-cut-bell-pair-benchmarking](https://quantum.cloud.ibm.com/docs/en/tutorials/edc-cut-bell-pair-benchmarking)

**Tahmini QPU kullanimi:** Heron r2 islemcide 22 saniye (yalnizca tahmindir).

<a id="bell-arka-plan"></a>
### Arka Plan

Kuantum donanimi tipik olarak yerel etkilesimlerle sinirlidir, ancak bircok algoritma uzak kubitleri ve hatta ayri islemcilerdeki kubitleri dolaniklastirmayi gerektirir. **Dinamik devreler** - yani devre ortasi olcum ve ileri besleme iceren devreler - gercek zamanli klasik iletisim kullanarak yerel olmayan kuantum islemlerini etkili bir sekilde uygulamanin bir yolunu saglar. Bu yaklasimda, bir devrenin (veya bir QPU'nun) bir bolumunden elde edilen olcum sonuclari, uzak kubitler arasinda dolanikligi teleporte etmemize olanak taniyan baska bir bolgedeki kapilari kosullu olarak tetikleyebilir. Bu, **yerel islemler ve klasik iletisim (LOCC)** semalarinin temelini olusturur.

LOCC'nin umut verici kullanim alanlarindan biri, teleportasyon yoluyla **sanal uzun menzilli CNOT kapilari** gerceklestirmektir. Dogrudan uzun menzilli bir CNOT yerine (ki donanim baglantisi buna izin vermeyebilir), Bell ciftleri olusturur ve teleportasyon tabanli bir kapi uygulamasi gerceklestiririz. Ancak, bu tur islemlerin sadakati donanim ozelliklerine baglidir. Kubit dekoheransi ve klasik iletisim gecikmesi dolanik durumu bozabilir.

Referans deneyinde, yazarlar her dort bagli kubit grubu uzerinde kucuk bir dinamik devre calistirarak, cihazin hangi bolumlerinin LOCC tabanli dolaniklama icin en uygun oldugunu belirlemek icin bir **Bell cifti sadakat karsilastirmasi** sunarlar.

Dort kubitlik devre su sekilde calisir:
- Kubitler 1 ve 2 yerel olarak bir **kesilmemis Bell cifti** haline getirilir (Hadamard ve CNOT ile).
- Teleportasyon rutini bu Bell ciftini tuketir ve kubitleri 0 ve 3'u LOCC kullanarak dolaniklastirir.
- Kubitler 1 ve 2, devrenin yurutulmesi sirasinda olculur ve bu sonuclara gore Pauli duzeltmeleri uygulanir (kubit 3'e X, kubit 0'a Z).
- Kubitleri 0 ve 3, devrenin sonunda bir Bell durumunda birakilir.

Son dolanik ciftin kalitesini olcmek icin **stabilizatorleri** olceriz: Z bazinda parite (Z0*Z3) ve X bazinda parite (X0*X3). Mukemmel bir Bell cifti icin her iki beklenti degeri de +1'e esittir.

**Ortalama Karesel Hata (MSE)** metrigi:

```
MSE = ((Z0*Z3 - 1)^2 + (X0*X3 - 1)^2) / 2
```

Daha dusuk MSE, kubit ciftinin ideale daha yakin bir Bell durumu elde ettigini gosterir.

<a id="bell-gereksinimler"></a>
### Gereksinimler ve Kurulum

**Gereksinimler:**
- Qiskit SDK v2.0 veya sonrasi (gorsellestirme destegi ile)
- Qiskit Runtime v0.40 veya sonrasi (`pip install qiskit-ibm-runtime`)

**Kurulum kodu:**

```python
from qiskit import QuantumCircuit

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler import generate_preset_pass_manager

import numpy as np
import matplotlib.pyplot as plt
```

**Temel fonksiyonlar:**

`create_bell_stab(initial_layouts)` - Her 1D kubit zinciri icin ZZ ve XX stabilizator olcum devreleri olusturur:

```python
def create_bell_stab(initial_layouts):
    """
    Create a circuit for a 1D chain of qubits (number of qubits must be a multiple of 4),
    where a middle Bell pair is consumed to create a Bell at the edge.
    Takes as input a list of lists, where each element of the list is a
    1D chain of physical qubits that is used as the initial_layout for the transpiled circuit.
    Returns a list of length-2 tuples, each tuple contains a circuit to measure the ZZ stabilizer and
    a circuit to measure the XX stabilizer of the edge Bell state.
    """
    bell_circuits = []
    for initial_layout in initial_layouts:
        assert (
            len(initial_layout) % 4 == 0
        ), f"The length of the chain must be a multiple of 4, len(inital_layout)={len(initial_layout)}"
        num_pairs = len(initial_layout) // 4

        bell_parallel = QuantumCircuit(4 * num_pairs, 4 * num_pairs)

        for pair_idx in range(num_pairs):
            (q0, q1, q2, q3) = (
                pair_idx * 4, pair_idx * 4 + 1,
                pair_idx * 4 + 2, pair_idx * 4 + 3,
            )
            (c0, c1) = pair_idx * 4, pair_idx * 4 + 3
            (ca0, ca1) = pair_idx * 4 + 1, pair_idx * 4 + 2

            bell_parallel.h(q0)
            bell_parallel.h(q1)
            bell_parallel.cx(q1, q2)
            bell_parallel.cx(q0, q1)
            bell_parallel.cx(q2, q3)
            bell_parallel.h(q2)

        bell_parallel.barrier()
        for pair_idx in range(num_pairs):
            (q0, q1, q2, q3) = (
                pair_idx * 4, pair_idx * 4 + 1,
                pair_idx * 4 + 2, pair_idx * 4 + 3,
            )
            (ca0, ca1) = pair_idx * 4 + 1, pair_idx * 4 + 2
            bell_parallel.measure(q1, ca0)
            bell_parallel.measure(q2, ca1)

        for pair_idx in range(num_pairs):
            (q0, q1, q2, q3) = (
                pair_idx * 4, pair_idx * 4 + 1,
                pair_idx * 4 + 2, pair_idx * 4 + 3,
            )
            (ca0, ca1) = pair_idx * 4 + 1, pair_idx * 4 + 2
            with bell_parallel.if_test((ca0, 1)):
                bell_parallel.x(q3)
            with bell_parallel.if_test((ca1, 1)):
                bell_parallel.z(q0)
                bell_parallel.id(q0)

        bell_zz = bell_parallel.copy()
        bell_zz.barrier()
        bell_xx = bell_parallel.copy()
        bell_xx.barrier()
        for pair_idx in range(num_pairs):
            (q0, q1, q2, q3) = (
                pair_idx * 4, pair_idx * 4 + 1,
                pair_idx * 4 + 2, pair_idx * 4 + 3,
            )
            bell_xx.h(q0)
            bell_xx.h(q3)
        bell_xx.barrier()
        for pair_idx in range(num_pairs):
            (q0, q1, q2, q3) = (
                pair_idx * 4, pair_idx * 4 + 1,
                pair_idx * 4 + 2, pair_idx * 4 + 3,
            )
            (c0, c1) = pair_idx * 4, pair_idx * 4 + 3
            bell_zz.measure(q0, c0)
            bell_zz.measure(q3, c1)
            bell_xx.measure(q0, c0)
            bell_xx.measure(q3, c1)

        bell_circuits.append(bell_zz)
        bell_circuits.append(bell_xx)

    return bell_circuits
```

`get_mse(result, initial_layouts)` - Sonuc nesnesinden her duzenleme icin MSE hesaplar:

```python
def get_mse(result, initial_layouts):
    """
    given a result object and the initial layouts, returns a dict of layouts and their mse
    """
    layout_mse = {}
    for layout_idx, initial_layout in enumerate(initial_layouts):
        layout_mse[tuple(initial_layout)] = {}
        num_pairs = len(initial_layout) // 4

        counts_zz = result[2 * layout_idx].data.c.get_counts()
        total_shots = sum(counts_zz.values())

        exp_zz_list = []
        for pair_idx in range(num_pairs):
            exp_zz = 0
            for bitstr, shots in counts_zz.items():
                bitstr = bitstr[::-1]
                b1, b0 = (bitstr[pair_idx * 4], bitstr[pair_idx * 4 + 3])
                z_val0 = 1 if b0 == "0" else -1
                z_val1 = 1 if b1 == "0" else -1
                exp_zz += z_val0 * z_val1 * shots
            exp_zz /= total_shots
            exp_zz_list.append(exp_zz)

        counts_xx = result[2 * layout_idx + 1].data.c.get_counts()
        total_shots = sum(counts_xx.values())

        exp_xx_list = []
        for pair_idx in range(num_pairs):
            exp_xx = 0
            for bitstr, shots in counts_xx.items():
                bitstr = bitstr[::-1]
                b1, b0 = (bitstr[pair_idx * 4], bitstr[pair_idx * 4 + 3])
                x_val0 = 1 if b0 == "0" else -1
                x_val1 = 1 if b1 == "0" else -1
                exp_xx += x_val0 * x_val1 * shots
            exp_xx /= total_shots
            exp_xx_list.append(exp_xx)

        mse_list = [
            ((exp_zz - 1) ** 2 + (exp_xx - 1) ** 2) / 2
            for exp_zz, exp_xx in zip(exp_zz_list, exp_xx_list)
        ]

        for idx in range(num_pairs):
            layout_mse[tuple(initial_layout)][
                tuple(initial_layout[4 * idx : 4 * idx + 4])
            ] = mse_list[idx]

    return layout_mse
```

`plot_mse_ecdfs(layouts_mse, combine_layouts=False)` - MSE verilerinin kumulatif dagilim fonksiyonunu (CDF) cizer:

```python
def plot_mse_ecdfs(layouts_mse, combine_layouts=False):
    """
    Plot CDF of MSE data for multiple layouts. Optionally combine all data in a single CDF
    """
    if not combine_layouts:
        for initial_layout, layouts in layouts_mse.items():
            sorted_layouts = dict(sorted(layouts.items(), key=lambda item: item[1]))
            layout_list = list(sorted_layouts.keys())
            mse_list = np.asarray(list(sorted_layouts.values()))
            x = np.array(mse_list)
            y = np.arange(1, len(x) + 1) / len(x)
            x = np.insert(x, 0, x[0])
            y = np.insert(y, 0, 0)
            plt.plot(x, y, marker="x", linestyle="-", label=f"qubits: {initial_layout}")
            for xi, yi, q in zip(x[1:], y[1:], layout_list):
                plt.annotate([q[0], q[3]], (xi, yi), textcoords="offset points",
                             xytext=(5, -10), ha="left", fontsize=8)
    elif combine_layouts:
        all_layouts = {}
        all_initial_layout = []
        for initial_layout, layouts in layouts_mse.items():
            all_layouts.update(layouts)
            all_initial_layout += initial_layout
        sorted_layouts = dict(sorted(all_layouts.items(), key=lambda item: item[1]))
        layout_list = list(sorted_layouts.keys())
        mse_list = np.asarray(list(sorted_layouts.values()))
        x = np.array(mse_list)
        y = np.arange(1, len(x) + 1) / len(x)
        x = np.insert(x, 0, x[0])
        y = np.insert(y, 0, 0)
        plt.plot(x, y, marker="x", linestyle="-",
                 label=f"qubits: {sorted(list(set(all_initial_layout)))}")
        for xi, yi, q in zip(x[1:], y[1:], layout_list):
            plt.annotate([q[0], q[3]], (xi, yi), textcoords="offset points",
                         xytext=(5, -10), ha="left", fontsize=8)

    plt.xscale("log")
    plt.xlabel("Mean squared error of <ZZ> and <XX>")
    plt.ylabel("Cumulative distribution function")
    plt.title("CDF for different initial layouts")
    plt.grid(alpha=0.3)
    plt.show()
```

<a id="bell-adim-1"></a>
### Adim 1: Klasik Girdileri Kuantum Problemine Esleme

Ilk adim, cihazin topolojisine uygun olarak tum aday Bell-cifti baglantilarini karsilastirmak icin bir dizi kuantum devresi olusturmaktir. Cihaz baglantisi haritasinda dort kubitlik tum dogrusal bagli zincirleri programatik olarak arariz.

```python
service = QiskitRuntimeService()
backend = service.least_busy(operational=True)
```

Zincirler, cihaz grafikte acgozlu (greedy) arama yapan bir yardimci fonksiyon kullanilarak olusturulur. 16 kubitlik gruplarda (4 adet 4-kubitlik zincir) "seritler" dondurur:

```python
from itertools import chain
from collections import defaultdict

def stripes16_from_backend(backend):
    """
    Creates stripes of 16 qubits, four non-overlapping four-qubit chains,
    that cover as much of the coupling map as possible.
    """
    edges = backend.coupling_map.get_edges()
    graph = defaultdict(set)
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)

    qubits = sorted(graph)
    used = set()
    blocks = []

    for q in qubits:
        if q in used:
            continue
        def extend(path):
            if len(path) == 4:
                return path
            tip = path[-1]
            for nbr in sorted(graph[tip]):
                if nbr not in path and nbr not in used:
                    maybe = extend(path + [nbr])
                    if maybe:
                        return maybe
            return None

        block = extend([q])
        if block:
            blocks.append(block)
            used.update(block)

    stripes = [
        list(chain.from_iterable(blocks[i : i + 4]))
        for i in range(0, len(blocks) // 4 * 4, 4)
    ]
    leftovers = set(qubits) - set(chain.from_iterable(stripes))
    return stripes, leftovers

initial_layouts, leftover = stripes16_from_backend(backend)
```

Devre her zincir icin su adimlari gerceklestirir:
1. **Ortadaki Bell cifti hazirlama:** Kubit 1'e Hadamard ve kubit 1'den kubit 2'ye CNOT uygulayarak |Phi+> = (|00> + |11>)/sqrt(2) durumu olusturulur.
2. **Kenar kubitlerini dolaniklastirma:** Kubit 0'dan 1'e ve kubit 2'den 3'e CNOT uygulanir. Kubit 2'ye Hadamard uygulanir.
3. **Devre ortasi olcum ve ileri besleme:** Kubitler 1 ve 2 hesaplama bazinda olculur. Olcum sonuclarina gore: m1=1 ise kubit 3'e X kapisi; m2=1 ise kubit 0'a Z kapisi uygulanir.
4. **Bell cifti stabilizatorlerini olcme:** ZZ (ilk devre) ve XX (ikinci devre) stabilizatorleri olculur.

```python
circuits = create_bell_stab(initial_layouts)
circuits[-1].draw("mpl", fold=-1)
```

<a id="bell-adim-2"></a>
### Adim 2: Kuantum Donanimi Icin Optimizasyon

Devreleri gercek donanim uzerinde calistirmadan once transpile edilmesi gerekir. Her zincir icin belirli fiziksel kubitler secilmis oldugundan, `optimization_level=0` ile sabit duzenleme kullanilir:

```python
isa_circuits = []
for ind, init_layout in enumerate(initial_layouts):
    pm = generate_preset_pass_manager(
        optimization_level=0, backend=backend, initial_layout=init_layout
    )
    isa_circ = pm.run(circuits[ind * 2 : ind * 2 + 2])
    isa_circuits.extend(isa_circ)
```

<a id="bell-adim-3"></a>
### Adim 3: Qiskit Primitives ile Calistirma

Deney, Qiskit Runtime ve Sampler primitive'i kullanilarak yurutulur:

```python
sampler = Sampler(mode=backend)
sampler.options.environment.job_tags = ["cut-bell-pair-test"]
job = sampler.run(isa_circuits)
```

<a id="bell-adim-4"></a>
### Adim 4: Sonuclarin Islenmesi ve Gorsellestirme

Her test edilen kubit grubu icin MSE metrigi hesaplanir. Sonuclar, cihaz genelinde genis bir dolaniklama kalitesi yelpazesi ortaya koyar ve makalede belirtilen bulguyu dogrular: Bell durumu sadakatinde fiziksel kubitlere bagli olarak **bir buyukluk mertebesinden fazla fark** olabilir.

```python
layouts_mse = get_mse(job.result(), initial_layouts)
```

**Ornek Sonuclar:**

| Layout | Kubitler | MSE |
|--------|----------|-----|
| [0-15] | [0, 1, 2, 3] | 0.0312 |
| [0-15] | [4, 5, 6, 7] | 0.0491 |
| [0-15] | [8, 9, 10, 11] | 0.0711 |
| [0-15] | [12, 13, 14, 15] | 0.0436 |
| [16-35] | [16, 23, 22, 21] | 0.0197 |
| [16-35] | [17, 27, 26, 25] | 0.113 |
| [56-75] | [56, 63, 62, 61] | **0.8663** |
| [116-135] | [117, 125, 126, 127] | **0.7246** |
| [136-155] | [137, 147, 146, 145] | **1.0187** |

CDF grafigi ile gorsellestirme:

```python
plot_mse_ecdfs(layouts_mse, combine_layouts=True)
```

CDF grafigi, x ekseninde MSE esigini ve y ekseninde bu MSE'ye sahip veya daha dusuk kubit ciftlerinin oranini gosterir. Dusuk MSE yakininda dik bir yukselis, bircok ciftin yuksek sadakatli oldugunu gosterir.

### Referanslar

[1] Carrera Vazquez, A., Tornow, C., Riste, D. et al. *Combining quantum processors with real-time classical communication.* Nature 636, 75-79 (2024).

---

## Bolum 4: Hamiltonian Simulasyonu Icin Derleme Yontemleri

**Orijinal Sayfa:** [https://quantum.cloud.ibm.com/docs/en/tutorials/compilation-methods-for-hamiltonian-simulation-circuits](https://quantum.cloud.ibm.com/docs/en/tutorials/compilation-methods-for-hamiltonian-simulation-circuits)

**Tahmini QPU kullanimi:** Bu egitimde yurutme yapilmamistir cunku transpilasyon surecine odaklanilmaktadir.

<a id="ham-arka-plan"></a>
### Arka Plan

Kuantum devresi derlemesi, kuantum hesaplama is akisindaki kritik bir adimdir. Ust duzey bir kuantum algoritmasini, hedef kuantum donaniminin kisitlamalarina uyan fiziksel bir kuantum devresine donusturmeyi icerir. Etkili derleme, devre derinligini, kapi sayisini ve yurutme suresini azaltarak kuantum algoritmalarinin performansini onemli olcude etkileyebilir.

Bu egitim, Qiskit'teki uc farkli devre derleme yaklasimini pratik orneklerle incelemektedir.

**Bu egitimde ogrenilecekler:**
- SABRE ile Qiskit transpiler'in duzenleme ve yonlendirme optimizasyonu icin nasil kullanilacagi
- AI transpiler'in gelismis, otomatik devre optimizasyonu icin nasil kullanilacagi
- Rustiq plugin'inin ozellikle Hamiltonian simulasyonu gorevlerinde hassas sentez icin nasil kullanilacagi

<a id="ham-yontemler"></a>
### Derleme Yontemlerine Genel Bakis

#### 1. SABRE ile Qiskit Transpiler

Qiskit transpiler, devre duzenleme ve yonlendirme optimizasyonu icin **SABRE (SWAP-based BidiREctional heuristic search)** algoritmasini kullanir. SABRE, donanim baglanti kisitlamalarina uyarken SWAP kapilarini ve bunlarin devre derinligi uzerindeki etkisini en aza indirmeye odaklanir. Bu yontem son derece cok yonludur ve genel amacli devre optimizasyonu icin uygundur.

#### 2. AI Transpiler

Qiskit'teki AI destekli transpiler, devre yapisi ve donanim kisitlamalarindaki kaliplari analiz ederek en uygun transpilasyon stratejilerini tahmin etmek icin **makine ogrenimi** kullanir. Buyuk olcekli kuantum devreleri icin ozellikle etkilidir. `AIPauliNetworkSynthesis` gecisi ile **Pauli ag devreleri** icin pekistirmeli ogrenme tabanli bir sentez yaklasimi uygulayabilir.

#### 3. Rustiq Plugin

Rustiq plugin'i, ozellikle Trotterize dinamiklerde kullanilan Pauli rotasyonlarini temsil eden **PauliEvolutionGate** islemleri icin gelismis sentez teknikleri sunar. Hamiltonian simulasyonu uygulayan devrelerde degerlidir ve hassas, dusuk derinlikli devre sentezi saglar.

### Gereksinimler

- Qiskit SDK v1.3 veya sonrasi
- Qiskit Runtime v0.28 veya sonrasi (`pip install qiskit-ibm-runtime`)
- Qiskit IBM Transpiler (`pip install qiskit-ibm-transpiler`)
- Qiskit AI Transpiler local mode (`pip install qiskit_ibm_ai_local_transpiler`)
- NetworkX (`pip install networkx`)

### Kurulum

```python
from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
from qiskit_ibm_transpiler import generate_ai_pass_manager
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
from collections import Counter
from IPython.display import display
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import requests
import logging

logging.getLogger("qiskit_ibm_transpiler.wrappers.ai_local_synthesis").setLevel(logging.ERROR)

seed = 42  # Tekrarlanabilirlik icin seed
```

<a id="ham-su2"></a>
### Bolum 1: Efficient SU2 Devresi

#### Adim 1: Devreyi Olusturma

Degisken kuantum algoritmalarinda (VQE gibi) ve kuantum makine ogrenimi gorevlerinde yaygin olarak kullanilan `efficient_su2` devresi incelenir. Dairesel bir desende duzenlenmis alternatif tek-kubit rotasyonlari ve dolaniklastirma kapilarindan olusur.

```python
qubit_size = list(range(10, 101, 10))
qc_su2_list = [
    efficient_su2(n, entanglement="circular", reps=1)
    .decompose()
    .copy(name=f"SU2_{n}")
    for n in qubit_size
]
qc_su2_list[0].draw(output="mpl")
```

#### Adim 2: Donanim Icin Optimizasyon

Uc derleme yonteminin konfigurasyonu:

**SABRE transpiler:**
```python
service = QiskitRuntimeService(channel="ibm_quantum_platform")
backend = service.backend("ibm_torino")

pm_sabre = generate_preset_pass_manager(
    optimization_level=3, backend=backend, seed_transpiler=seed
)
```

**AI transpiler:**
```python
pm_ai = generate_ai_pass_manager(
    backend=backend, optimization_level=3, ai_optimization_level=3
)
```

**Rustiq plugin:**
```python
hls_config = HLSConfig(
    PauliEvolution=[
        (
            "rustiq",
            {
                "nshuffles": 400,
                "upto_phase": True,
                "fix_clifford": True,
                "preserve_order": False,
                "metric": "depth",
            },
        )
    ]
)
pm_rustiq = generate_preset_pass_manager(
    optimization_level=3,
    backend=backend,
    hls_config=hls_config,
    seed_transpiler=seed,
)
```

#### Metrikleri Yakalama Fonksiyonu

Derleme yontemlerinin performansini karsilastirmak icin transpile edilen devrenin toplam derinligini, genel kapi sayisini, transpilasyon suresini ve **2-kubit kapi derinligini** kaydeden bir fonksiyon tanimlanir:

```python
def capture_transpilation_metrics(results, pass_manager, circuits, method_name):
    transpiled_circuits = []
    for i, qc in enumerate(circuits):
        start_time = time.time()
        transpiled_qc = pass_manager.run(qc)
        end_time = time.time()
        transpiled_qc = transpiled_qc.decompose(gates_to_decompose=["swap"])
        transpilation_time = end_time - start_time
        circuit_depth = transpiled_qc.depth(lambda x: x.operation.num_qubits == 2)
        circuit_size = transpiled_qc.size()
        results.loc[len(results)] = {
            "method": method_name, "qc_name": qc.name, "qc_index": i,
            "num_qubits": qc.num_qubits, "ops": transpiled_qc.count_ops(),
            "depth": circuit_depth, "size": circuit_size, "runtime": transpilation_time,
        }
        transpiled_circuits.append(transpiled_qc)
    return transpiled_circuits
```

#### SU2 Sonuclari

| Yontem | Ort. Derinlik | Ort. Boyut | Ort. Sure (s) |
|--------|---------------|------------|---------------|
| **AI** | **56.4** | 852.5 | 45.89 |
| SABRE | 64.6 | 864.9 | 52.57 |

**Analiz:**
- Ortalamada AI transpiler devre derinligi acisindan daha iyi performans gosterir (%10'dan fazla iyilesme).
- Kapi sayisi ve transpilasyon suresi icin her iki yontem benzer sonuclar verir.
- Ozellikle 30, 50, 70 ve 90 kubitlik devrelerde AI transpiler, SABRE'den onemli olcude daha sig devreler bulmustur.
- **Onemli cikarim:** SABRE ve AI genellikle karsilastirabilir sonuclar uretirken, AI transpiler ozellikle derinlik acisindan ara sira cok daha iyi cozumler kesfedebilir.

<a id="ham-hamiltonian"></a>
### Bolum 2: Hamiltonian Simulasyon Devresi

#### Adim 1: PauliEvolutionGate ile Devreleri Inceleme

Bu bolumde, Hamiltonian'larin verimli simulasyonunu saglayan `PauliEvolutionGate` kullanilarak olusturulan kuantum devreleri incelenmektedir.

**Kullanilan Hamiltonian'lar:** ZZ, XX ve YY gibi ciftler arasi etkilesimleri tanimlayan Hamiltonian'lar. Bunlar kuantum kimyasi, yogun madde fizigi ve malzeme biliminde yaygin olarak kullanilir.

**Kaynak:** Hamlib benchmark deposundan ve Benchpress cercevesinden alinan devreler.

```python
url = "https://raw.githubusercontent.com/Qiskit/benchpress/e7b29ef7be4cc0d70237b8fdc03edbd698908eff/benchpress/hamiltonian/hamlib/100_representative.json"
response = requests.get(url)
ham_records = json.loads(response.text)
ham_records = [h for h in ham_records if h["ham_qubits"] <= backend.num_qubits]
ham_records = sorted(ham_records, key=lambda x: x["ham_terms"])[:35]

qc_ham_list = []
for h in ham_records:
    terms = h["ham_hamlib_hamiltonian_terms"]
    coeff = h["ham_hamlib_hamiltonian_coefficients"]
    num_qubits = h["ham_qubits"]
    name = h["ham_problem"]
    evo_gate = PauliEvolutionGate(SparsePauliOp(terms, coeff))
    qc_ham = QuantumCircuit(num_qubits)
    qc_ham.name = name
    qc_ham.append(evo_gate, range(num_qubits))
    qc_ham_list.append(qc_ham)
```

#### Adim 2: Transpilasyon ve Karsilastirma

Uc yontem de ayni backend uzerinde karsilastirilir:

```python
results_ham = pd.DataFrame(
    columns=["method", "qc_name", "qc_index", "num_qubits", "ops", "depth", "size", "runtime"]
)

tqc_sabre = capture_transpilation_metrics(results_ham, pm_sabre, qc_ham_list, "sabre")
tqc_ai = capture_transpilation_metrics(results_ham, pm_ai, qc_ham_list, "ai")
tqc_rustiq = capture_transpilation_metrics(results_ham, pm_rustiq, qc_ham_list, "rustiq")
```

#### Hamiltonian Sonuclari

| Yontem | Ort. Derinlik | Ort. Boyut | Ort. Sure (s) |
|--------|---------------|------------|---------------|
| AI | 316.86 | 2181.26 | 5.97 |
| **Rustiq** | **281.94** | 2268.80 | 3.86 |
| SABRE | 337.97 | **2120.14** | **3.07** |

#### En Iyi Yontem Analizi (Devre Bazinda)

**Derinlik icin en iyi yontem:**
- AI: 16 devre (%45.7)
- Rustiq: 16 devre (%45.7)
- SABRE: 10 devre (%28.6)

**Boyut icin en iyi yontem:**
- SABRE: 18 devre (%51.4)
- Rustiq: 14 devre (%40.0)
- AI: 10 devre (%28.6)

#### Hamiltonian Devre Derleme Sonuclarinin Analizi

- **Rustiq**, ortalamada devre derinligi acisindan en iyi performansi gostermis ve SABRE'den yaklasik %20 daha dusuk derinlik elde etmistir. Bu beklenen bir sonuctur cunku Rustiq, PauliEvolutionGate islemlerini optimize edilmis, dusuk derinlikli ayristirma stratejileriyle sentezlemek icin ozel olarak tasarlanmistir.

- **AI transpiler**, devre derinligi icin guclu ve tutarli performans gostermis, cogu devrede SABRE'yi geride birakmistir. Ancak, ozellikle buyuk devrelerde en yuksek calisma suresine neden olmustur.

- **SABRE**, en yuksek ortalama derinlige sahip olmasina ragmen, en dusuk ortalama kapi sayisini elde etmistir. Bu, SABRE'nin sezgiselinin dogrudan kapi sayisini en aza indirmeye oncelik vermesiyle uyumludur.

<a id="ham-ozet"></a>
### Ozet ve Oneriler

AI transpiler genellikle SABRE'den daha iyi sonuclar verse de, cikarim basitce "her zaman AI transpiler kullan" olmamalidir. Dikkate alinmasi gereken onemli nanslar vardir:

1. **AI transpiler** genellikle guvenilirdir ve derinlik-optimize edilmis devreler saglar, ancak calisma suresi, desteklenen baglanti haritalari ve sentez yetenekleri acisindan odulesimler icerir.

2. **SABRE transpiler** son derece guvenilir olmaya devam eder ve parametreleri ayarlanarak (ornegin `layout_trials=400`, `swap_trials=400`) daha da optimize edilebilir.

3. **Rustiq**, PauliEvolutionGate iceren devreler icin ozel olarak tasarlanmistir ve Hamiltonian simulasyon problemleri icin genellikle en iyi performansi verir.

**Oneri:** Herkese uyan tek bir transpilasyon stratejisi yoktur. Kullanicilar, devrelerinin yapisini anlamali ve belirli problemleri ve donanim kisitlamalari icin en verimli cozumu bulmak uzere birden fazla transpilasyon yontemini - AI, SABRE ve Rustiq gibi ozellestirilmis araclar dahil - test etmeye tesvik edilir.

### Referanslar

[1] H. Zou, M. Treinish, K. Hartman, A. Ivrii, J. Lishman et al. *"LightSABRE: A Lightweight and Enhanced SABRE Algorithm"*. [arXiv:2409.08368](https://arxiv.org/abs/2409.08368)

[2] D. Kremer, V. Villar, H. Paik, I. Duran, I. Faro, J. Cruz-Benito et al. *"Practical and efficient quantum circuit synthesis and transpiling with Reinforcement Learning"*. [arXiv:2405.13196](https://arxiv.org/abs/2405.13196)

[3] A. Dubal, D. Kremer, S. Martiel, V. Villar, D. Wang, J. Cruz-Benito et al. *"Pauli Network Circuit Synthesis with Reinforcement Learning"*. [arXiv:2503.14448](https://arxiv.org/abs/2503.14448)

[4] T. Goubault de Brugiere, S. Martiel et al. *"Faster and shorter synthesis of Hamiltonian simulation circuits"*. [arXiv:2404.03280](https://arxiv.org/abs/2404.03280)

---

---

## Bolum 5: Proje Deneyimleri — Kodda Yasanan Problemler ve Cozumleri

> Bu bolum, quantum-attention projesinde yasanan gercek problemleri ve cozumlerini icerir.
> Gelecekte bu projeyle calisan herhangi bir LLM veya gelistirici icin referans.
> Tarih: 2026-03-08

### 5.1 Qiskit 2.x API Degisiklikleri

**Problem:** `QiskitRuntimeService.save_account(channel='ibm_quantum')` calismadi.
**Hata:** `InvalidAccountError: "Invalid channel value... got 'ibm_quantum'"`
**Sebep:** Qiskit 2.x (2025+) channel adini `ibm_quantum` → `ibm_quantum_platform` olarak degistirdi.
**Cozum:**
```python
QiskitRuntimeService.save_account(
    channel='ibm_quantum_platform',  # DOGRU
    token='YOUR_TOKEN',
    overwrite=True
)
```
**Ders:** IBM Quantum dokumantasyonu hizla degisiyor. Qiskit 1.x ornekleri artik calismayabilir. Her zaman versiyon kontrolu yap.

### 5.2 Deprecated Class-Based API

**Problem:** `ZZFeatureMap()` ve `EfficientSU2()` DeprecationWarning veriyor.
**Sebep:** Qiskit 2.1+ class-based API'yi deprecated etti.
**Dogru kullanim:**
```python
# ESKI (deprecated):
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
fm = ZZFeatureMap(8)

# YENI (fonksiyon API):
from qiskit.circuit.library import zz_feature_map, efficient_su2
fm = zz_feature_map(8)
```
**Not:** Simdlik class-based hala calisiyor ama gelecek surumde kaldirabilirler.

### 5.3 Parameter-Shift Rule Performance Problemi

**Problem:** Qiskit EstimatorQNN ile quantum model training pratik olarak kullanilamaz derecede yavas.
**Olcumler:**
- Forward pass: 18ms/sample (kabul edilebilir)
- Backward pass: 1501ms/sample (KABUL EDILEMEZ)
- Oran: 83x (backward/forward)
- Tahmini toplam training: 285 saat (22500 sample, 30 epoch)

**Sebep:** Parameter-shift rule:
```
f'(theta) = [f(theta + pi/2) - f(theta - pi/2)] / 2
```
Her parametrenin gradyani icin 2 ek circuit evaluation gerekir. 48 parametre = 96 ek evaluation/sample.

**Cozum:** PyTorch-native statevector simulasyonu (quantum_attention_fast.py):
- Gate'ler = PyTorch tensor islemleri (cos, sin, matmul)
- Gradient = autograd chain rule (tek backward pass)
- Sonuc: 14ms/sample backward (107x hizlanma)
- Toplam training: 5-14 dakika (3400x hizlanma)

**Trade-off:** PyTorch-native sadece simulasyonda calisir. Gercek QPU'da parameter-shift veya SPSA gerekir.
**Tavsiye:** Egitimi simulatorde PyTorch ile yap, QPU'da sadece inference calistir. Parametreleri transfer et.

### 5.4 Encoding-Preprocessing Uyumu (KRITIK BULGU)

**Problem:** Quantum model %52 accuracy verdi — random tahmin seviyesi.
**Teshis:** PCA ciktilari ~N(0,1) dagiliminda, aralik yaklasik [-3, +3]. Ama angle encoding Ry(x) icin optimal aralik [0, pi]:
- Ry(0) = |0> durumu
- Ry(pi) = |1> durumu
- Ry(negatif deger) = Ry(pozitif deger ile ayni (periodiklik))
- Aralik uyumsuzlugu → circuit bilgi kaybi

**Cozum:**
```python
x = torch.sigmoid(x) * math.pi  # N(0,1) → (0, pi)
```

**Sonuc:** %52 → %69 (+17 puan!)
**Ders:** Kuantum ML'de encoding araligi ile input veri araligi uyumlu OLMAK ZORUNDA. Bu, klasik ML'deki normalization/standardization'in kuantum karsiligi. Yanlislik istatistiksel olarak gizli — model "ogreniyor gibi" gorunur ama aslinda bilgi tasiyamaz.

### 5.5 IBM QPU Bellek Limiti (Error 6073)

**Problem:** IBM QPU'da inference calistirirken job fail oldu.
**Hata:** `Error 6073 — The size of the job exceeds the memory limits`
**Sebep:** 50 sample x 8 observable = 400 PUB (Primitive Unified Bloc) tek job'da gonderildi. IBM QPU'larin klasik kontrol donanimi (FPGA/ASIC) 400 ayri circuit bind + observable evaluation icin yeterli bellege sahip degil.

**Cozum:**
```python
# 5'erli batch'ler (5 sample x 8 obs = 40 PUB/job — guvenli)
batch_size = 5
for i in range(0, n_samples, batch_size):
    batch_pubs = pubs[i*8 : (i+batch_size)*8]
    job = estimator.run(batch_pubs)
    results.append(job.result())
```

**Guvenli PUB limitleri (deneyimsel):**
- < 50 PUB: guvenli
- 50-100 PUB: genellikle calisir
- 100-200 PUB: risk baslar
- 400+ PUB: kesinlikle fail

**Ders:** NISQ caginda bottleneck sadece kuantum islemci degil — klasik kontrol donanimi, kuyruk sistemi ve bellek de sinirlamalara sahip. "Hybrid quantum-classical" teriminin gercek anlami: her iki tarafin da sinirlamalari var.

### 5.6 HuggingFace Dataset + PyTorch Tensor Uyumsuzlugu

**Problem:** HuggingFace `Dataset` nesnesi PyTorch tensor ile indexlenemiyor.
**Hata:** `TypeError: len() of a 0-d tensor`
**Sebep:** `dataset[tensor_index]` cagrisi 0-boyutlu tensor'u int olarak yorumlayamiyor.
**Cozum:**
```python
# YANLIS:
subset = dataset[perm[:1000]]

# DOGRU:
subset = dataset[perm[:1000].tolist()]
```
**Ders:** Framework'ler arasi veri gecislerinde tip donusumune dikkat. PyTorch tensor → Python list → HuggingFace index.

### 5.7 PyTorch 2.6 weights_only Default Degisikligi

**Problem:** `torch.load()` numpy array iceren dosyalari yukleyemiyor.
**Hata:** `UnpicklingError` (weights_only=True default)
**Sebep:** PyTorch 2.6 guvenlik icin `weights_only=True` default yapti. Bu, numpy array gibi non-tensor objeleri reddediyor.
**Cozum:**
```python
data = torch.load(path, weights_only=False)  # Kendi dosyamiz, guvenli
```
**Uyari:** Guvenilmeyen kaynaklardan gelen dosyalarda `weights_only=True` kullanin. `False` sadece kendi olusturdugumuz dosyalar icin.

### 5.8 Identity Initialization ve CZ Gate'ler

**Problem:** Identity-initialized circuit'in output'u |0...0> bekleniyordu ama farkli sonuc verdi.
**Sebep:** CZ (controlled-Z) gate'ler parametresiz — her zaman entanglement yapar.
- Ry(0) = I (identity), Rz(0) = I → dogru
- CZ gate: |11> → -|11> (faz degisikligi) → her zaman aktif
- Yani "identity init" sadece trainable parametrelerin sifir olmasi demek, tum circuit'in I olmasi degil.

**Ders:** "Identity initialization" literaturde yaniltici olabilir. EfficientSU2'de tam identity elde etmek icin CZ gate'leri de kontrol etmek gerekir (mumkun degil — sabit yapi). Dogru terim: "zero-parameter initialization" olmali.

### 5.9 Windows ve Unicode Kodlama

**Problem:** Python print statement'lari Unicode karakterlerle crash ediyor.
**Cozum:** Her script calistirmada:
```bash
PYTHONIOENCODING=utf-8 python script.py
```
Veya script icinde:
```python
os.environ["PYTHONIOENCODING"] = "utf-8"
```
**Ders:** Windows terminal'i default olarak cp1252 encoding kullanir. Turkce karakterler ve ozel semboller (⟨⟩, →, ≥) sorun cikarir.

### 5.10 Genel Tavsiyeler (Gelecek Projeler Icin)

1. **Kuantum ML projesine baslarken:** ONCE klasik baseline'i kur ve egit. Sonra kuantum modeli ayni kosullarda test et. Baseline yoksa "kuantum calisiyor" iddiasi anlamsiz.
2. **Encoding secimi:** Input veri araligini ve gate'in beklenen araligini MUTLAKA kontrol et. Uyumsuzluk = ogrenme yok.
3. **Qiskit vs PyTorch:** Qiskit simulasyonda cok yavas (parameter-shift rule). Training icin PyTorch-native kullan, Qiskit sadece QPU inference icin.
4. **IBM QPU:** Job boyutunu kucuk tut (<50 PUB). Batch'le. Queue suresi degisken (30s - 10dk).
5. **Reproducibility:** Seed MUTLAKA ayarla (torch, numpy, random). Her deneyin config'ini JSON olarak kaydet.
6. **Git kullan:** Her milestone'da commit at. Quantum ML'de "bir sey bozuldu, geri donelim" sik yasanir.

> **Not:** Bu belge, IBM Quantum Platform dokumantasyonundan 2026-03-08 tarihinde otomatik olarak derlenistir. En guncel bilgiler icin orijinal sayfalara basvurunuz.
> (c) IBM Corp., 2017-2026

---

## Phase 8 QPU Run: Balanced T1 Bias Validation (2026-03-11)

### Amac
T1 amplitude damping bias'ini test etmek icin dengeli (5 Label 0 + 5 Label 1) test seti ile QPU inference.

### Baglanti Detaylari
- **Token:** Yeni token kullanildi (2. IBM hesabi)
- **Channel:** `ibm_quantum_platform`
- **Backend:** ibm_fez (156 qubit, Eagle r3)
- **DNS Sorunu:** `globalcatalog.cloud.ibm.com` DNS resolution intermittent fail. Cozum: retry. Windows DNS cache flush islevsiz olabiliyor bash'te (cmd.exe /c "ipconfig /flushdns" gerekli).

### Optimizasyon
- **batch_size=10** (Phase 7: 5) → Her encoding tek job'da tamamlandi
- **PUB limiti:** 80 PUB/job (8 qubit x 10 sample) guvenli calisti. IBM Open Plan limiti >80 PUB.
- **Toplam job sayisi:** 4 (Phase 7: 8) — %50 azalma
- **Fallback mekanizmasi:** Kodda 80 PUB basarisiz olursa otomatik 40 PUB'a (batch_size=5) donuyor

### Job Detaylari
| Job ID | Encoding | PUB | Queue+Runtime |
|--------|----------|-----|---------------|
| d6ogisu9td6c73apbbh0 | Angle | 80 | 393.8s |
| d6ogluu9td6c73apbec0 | Dense | 40 | 485.7s |
| d6ogpom9td6c73apbi40 | IQP | 80 | 539.0s |
| d6ogtv43pels73a2q6g0 | Reupload | 80 | 144.0s |

### Sonuclar
TUM 4 encoding icin %100 sim-QPU agreement, sifir T1 bias.
L0 agreement = L1 agreement = %100 her encoding icin.

### Ogrenimler
1. **80 PUB guvenli:** IBM Open Plan'da 80 PUB tek job hatasiz calisti
2. **T1 bias bu devre derinliklerinde gorulmuyor:** depth 18-101 arasinda T1 etkisi tespit edilmedi
3. **Kota yonetimi:** Farkli IBM token'larla taze kota alinabilir (ayri hesap)
4. **Python buffering:** Background task'larda `PYTHONUNBUFFERED=1` eklenmeli, yoksa stdout gorünmuyor
