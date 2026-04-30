"""
Data Preprocessing Pipeline
============================
1. IMDb ve SST-2 datasetlerini HuggingFace'den indir
2. DistilBERT (frozen) ile 768-dim embedding cikart
3. PCA ile 768 -> 8 boyuta indirge
4. Train/val/test split'leri .pt dosyalarina kaydet

Kullanim:
    PYTHONIOENCODING=utf-8 python data/preprocess.py

Cikti:
    data/imdb_embeddings.pt   — {train_X, train_y, val_X, val_y, test_X, test_y, pca_variance}
    data/sst2_embeddings.pt   — ayni format
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Windows encoding fix
os.environ["PYTHONIOENCODING"] = "utf-8"

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def download_datasets():
    """HuggingFace'den IMDb ve SST-2 indir."""
    from datasets import load_dataset

    print("=" * 60)
    print("ADIM 1: Dataset Indirme")
    print("=" * 60)

    # IMDb: 25K train + 25K test, binary sentiment
    t0 = time.time()
    print("\n[1/2] IMDb indiriliyor (50K review)...")
    imdb = load_dataset("imdb")
    dt = time.time() - t0
    print(f"  -> IMDb yuklendi: train={len(imdb['train'])}, test={len(imdb['test'])}")
    print(f"  -> Sure: {dt:.1f}s")

    # SST-2: GLUE benchmark sentiment, train/validation/test
    t0 = time.time()
    print("\n[2/2] SST-2 indiriliyor (GLUE)...")
    sst2 = load_dataset("glue", "sst2")
    dt = time.time() - t0
    print(f"  -> SST-2 yuklendi: train={len(sst2['train'])}, val={len(sst2['validation'])}")
    print(f"  -> Sure: {dt:.1f}s")

    return imdb, sst2


def extract_embeddings(texts, model, tokenizer, device, batch_size=64, desc=""):
    """
    DistilBERT ile CLS token embedding'lerini cikart.

    DistilBERT nasil calisir:
    - Her cumleyi tokenize eder (kelime parcalarina ayirir)
    - Token'lari 768 boyutlu vektorlere donusturur
    - 6 transformer katmanindan gecirir
    - [CLS] token'inin son katman output'u = cumlenin ozet vektoru (768-dim)
    - Bu vektor sentiment, anlam, yapi bilgisini tasir

    Neden frozen (gradyan yok)?
    - DistilBERT zaten 66M parametre ile pre-trained
    - Biz sadece sabit feature extractor olarak kullaniyoruz
    - Trainable olan kisim: PCA sonrasi quantum circuit + linear head
    """
    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():  # Gradyan hesaplamasi KAPALI — frozen model
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_idx = i // batch_size + 1

            if batch_idx % 50 == 1 or batch_idx == n_batches:
                print(f"  {desc} Batch {batch_idx}/{n_batches}...")

            # Tokenize: metin -> token ID'leri + attention mask
            encoded = tokenizer(
                batch_texts,
                padding=True,          # Kisa cumleleri pad'le
                truncation=True,       # 512 token'dan uzunlari kes
                max_length=128,        # Max 128 token (hiz icin)
                return_tensors="pt",   # PyTorch tensor dondur
            ).to(device)

            # Forward pass: token ID -> 768-dim embedding
            outputs = model(**encoded)

            # [CLS] token embedding'i al (index 0)
            # outputs.last_hidden_state shape: (batch, seq_len, 768)
            # [:, 0, :] = her ornekteki [CLS] token = cumle ozeti
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(cls_embeddings.cpu())

    # Tum batch'leri birlestir: (N, 768)
    return torch.cat(all_embeddings, dim=0)


def apply_pca(train_X, val_X, test_X, n_components=8):
    """
    PCA ile 768 -> 8 boyut indirgeme.

    Neden PCA?
    - 768 boyutlu vektor 768 qubit gerektirir (imkansiz)
    - PCA en yuksek varyans'i tutan ilk 8 bileseni secer
    - Bilgi kaybi var ama olculebilir (explained_variance_ratio)
    - PCA SADECE train verisiyle fit edilir (data leakage onleme!)

    Neden 8 boyut?
    - 8 qubit = 2^8 = 256 kuantum durumu (simulasyonda yonetilabilir)
    - IBM QPU'da 8 qubit trivial (133-156 qubit mevcut)
    - Daha az qubit = daha az barren plateau riski
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    print("\n" + "=" * 60)
    print(f"ADIM 3: PCA 768 -> {n_components} Boyut Indirgeme")
    print("=" * 60)

    t0 = time.time()

    # Once standardize et (ortalama=0, std=1) — PCA icin onemli
    scaler = StandardScaler()
    train_np = scaler.fit_transform(train_X.numpy())  # Train ile fit
    val_np = scaler.transform(val_X.numpy())           # Val/test sadece transform
    test_np = scaler.transform(test_X.numpy())

    # PCA fit (sadece train verisiyle!)
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_np)
    val_pca = pca.transform(val_np)
    test_pca = pca.transform(test_np)

    dt = time.time() - t0

    # Aciklanan varyans orani — ne kadar bilgi korundu?
    var_ratio = pca.explained_variance_ratio_
    total_var = var_ratio.sum()

    print(f"  -> PCA tamamlandi ({dt:.2f}s)")
    print(f"  -> Korunan varyans: {total_var:.4f} ({total_var*100:.1f}%)")
    print(f"  -> Bilesen bazinda: {[f'{v:.3f}' for v in var_ratio]}")
    print(f"  -> Boyut: {train_X.shape[1]} -> {n_components}")

    return (
        torch.tensor(train_pca, dtype=torch.float32),
        torch.tensor(val_pca, dtype=torch.float32),
        torch.tensor(test_pca, dtype=torch.float32),
        var_ratio,
    )


def process_imdb(imdb, model, tokenizer, device):
    """IMDb dataset'ini isle: embedding + PCA."""
    print("\n" + "=" * 60)
    print("ADIM 2a: IMDb Embedding Cikartma (DistilBERT frozen)")
    print("=" * 60)

    # IMDb'de train/test var, validation yok -> train'den ayir
    train_texts = imdb["train"]["text"]
    test_texts = imdb["test"]["text"]
    train_labels = torch.tensor(imdb["train"]["label"], dtype=torch.long)
    test_labels = torch.tensor(imdb["test"]["label"], dtype=torch.long)

    print(f"  Train: {len(train_texts)} ornek, Test: {len(test_texts)} ornek")

    # Train embedding
    t0 = time.time()
    print("\n  [Train] Embedding cikartiliyor...")
    train_emb = extract_embeddings(train_texts, model, tokenizer, device, desc="[Train]")
    dt_train = time.time() - t0
    print(f"  -> Train embedding: {train_emb.shape}, sure: {dt_train:.1f}s")

    # Test embedding
    t0 = time.time()
    print("\n  [Test] Embedding cikartiliyor...")
    test_emb = extract_embeddings(test_texts, model, tokenizer, device, desc="[Test]")
    dt_test = time.time() - t0
    print(f"  -> Test embedding: {test_emb.shape}, sure: {dt_test:.1f}s")

    # Train'den %10 validation ayir (seed=42 reproducibility)
    n_val = len(train_emb) // 10  # 2500
    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(train_emb), generator=gen)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    val_emb = train_emb[val_idx]
    val_labels = train_labels[val_idx]
    train_emb = train_emb[train_idx]
    train_labels = train_labels[train_idx]

    print(f"\n  Split: train={len(train_emb)}, val={len(val_emb)}, test={len(test_emb)}")

    # PCA
    train_pca, val_pca, test_pca, var_ratio = apply_pca(train_emb, val_emb, test_emb)

    # Kaydet
    save_path = DATA_DIR / "imdb_embeddings.pt"
    torch.save(
        {
            "train_X": train_pca,
            "train_y": train_labels,
            "val_X": val_pca,
            "val_y": val_labels,
            "test_X": test_pca,
            "test_y": test_labels,
            "pca_variance": var_ratio,
            "original_dim": 768,
            "reduced_dim": 8,
        },
        save_path,
    )
    print(f"\n  -> Kaydedildi: {save_path} ({save_path.stat().st_size / 1024:.0f} KB)")

    return train_pca, val_pca, test_pca


def process_sst2(sst2, model, tokenizer, device):
    """SST-2 dataset'ini isle: embedding + PCA."""
    print("\n" + "=" * 60)
    print("ADIM 2b: SST-2 Embedding Cikartma (DistilBERT frozen)")
    print("=" * 60)

    # SST-2: train + validation (test label'lari yok, -1)
    train_texts = sst2["train"]["sentence"]
    val_texts = sst2["validation"]["sentence"]
    train_labels = torch.tensor(sst2["train"]["label"], dtype=torch.long)
    val_labels = torch.tensor(sst2["validation"]["label"], dtype=torch.long)

    print(f"  Train: {len(train_texts)} ornek, Val: {len(val_texts)} ornek")

    # Train'den %15'ini test olarak ayir (SST-2 test label'lari public degil)
    n_test = len(train_texts) // 7  # ~9600
    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(train_texts), generator=gen)

    test_idx = perm[:n_test]
    train_idx_subset = perm[n_test:]

    # HuggingFace Dataset tensor index kabul etmiyor, int listesine cevir
    test_idx_list = test_idx.tolist()
    train_idx_list = train_idx_subset.tolist()

    test_texts_split = [train_texts[i] for i in test_idx_list]
    test_labels = train_labels[test_idx]
    train_texts_split = [train_texts[i] for i in train_idx_list]
    train_labels_split = train_labels[train_idx_list]

    print(f"  Split: train={len(train_texts_split)}, val={len(val_texts)}, test={len(test_texts_split)}")

    # Train embedding
    t0 = time.time()
    print("\n  [Train] Embedding cikartiliyor...")
    train_emb = extract_embeddings(train_texts_split, model, tokenizer, device, desc="[Train]")
    dt = time.time() - t0
    print(f"  -> Train embedding: {train_emb.shape}, sure: {dt:.1f}s")

    # Val embedding
    t0 = time.time()
    print("\n  [Val] Embedding cikartiliyor...")
    val_emb = extract_embeddings(val_texts, model, tokenizer, device, desc="[Val]")
    dt = time.time() - t0
    print(f"  -> Val embedding: {val_emb.shape}, sure: {dt:.1f}s")

    # Test embedding
    t0 = time.time()
    print("\n  [Test] Embedding cikartiliyor...")
    test_emb = extract_embeddings(test_texts_split, model, tokenizer, device, desc="[Test]")
    dt = time.time() - t0
    print(f"  -> Test embedding: {test_emb.shape}, sure: {dt:.1f}s")

    # PCA
    train_pca, val_pca, test_pca, var_ratio = apply_pca(train_emb, val_emb, test_emb)

    # Kaydet
    save_path = DATA_DIR / "sst2_embeddings.pt"
    torch.save(
        {
            "train_X": train_pca,
            "train_y": train_labels_split,
            "val_X": val_pca,
            "val_y": val_labels,
            "test_X": test_pca,
            "test_y": test_labels,
            "pca_variance": var_ratio,
            "original_dim": 768,
            "reduced_dim": 8,
        },
        save_path,
    )
    print(f"\n  -> Kaydedildi: {save_path} ({save_path.stat().st_size / 1024:.0f} KB)")

    return train_pca, val_pca, test_pca


def main():
    print("=" * 60)
    print("Quantum Attention — Data Preprocessing Pipeline")
    print("=" * 60)
    total_t0 = time.time()

    # GPU/CPU secimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # 1. Dataset indirme
    imdb, sst2 = download_datasets()

    # 2. DistilBERT yukleme
    print("\n" + "=" * 60)
    print("DistilBERT Model Yukleniyor (frozen)")
    print("=" * 60)
    t0 = time.time()

    from transformers import DistilBertModel, DistilBertTokenizer

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval()  # Eval mode — dropout kapali
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    dt = time.time() - t0
    print(f"  -> DistilBERT yuklendi: {n_params/1e6:.1f}M parametre")
    print(f"  -> Sure: {dt:.1f}s")
    print(f"  -> Tum parametreler FROZEN (gradyan yok)")

    # 3. IMDb isleme
    process_imdb(imdb, model, tokenizer, device)

    # 4. SST-2 isleme
    process_sst2(sst2, model, tokenizer, device)

    total_dt = time.time() - total_t0
    print("\n" + "=" * 60)
    print(f"TUM ISLEMLER TAMAMLANDI — Toplam sure: {total_dt:.1f}s ({total_dt/60:.1f} dk)")
    print("=" * 60)


if __name__ == "__main__":
    main()
