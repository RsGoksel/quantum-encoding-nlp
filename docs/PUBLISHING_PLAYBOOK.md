# ResearchGate + GitHub Publishing Playbook

> **LLM kullanım kılavuzu.** Kullanıcı "paper hazır, yayınla" dediğinde bu playbook'u baştan sona uygula. Manuel müdahale gereken adımlar **🧑 USER** ile, otomatik adımlar **🤖 AGENT** ile işaretli.
>
> Bu playbook 2026-04-30 tarihinde `quantum-encoding-nlp` paper'ının ResearchGate'e yüklenmesi sürecinden çıkarılmıştır. DOI: `10.13140/RG.2.2.32211.13607`.

---

## Kullanıcı Kimliği (Sabit Bilgiler)

```yaml
name: Kadir Göksel Gündüz
email: gokssel.gunduz@gmail.com
orcid: 0009-0007-9120-5659
github_user: RsGoksel
affiliation: "Energy Institute, Istanbul Technical University (İTÜ), Istanbul, Türkiye"
language_preference: "Turkish (chat) + English (code/paper/commits)"
windows_path: "c:\\Users\\gokss\\OneDrive\\Masaüstü\\Kuantum\\<project>\\"
```

GitHub auth'u sistemde kayıtlı (RsGoksel). `gh auth status` ile doğrula.

---

## Ön Koşullar (Başlamadan Önce)

🤖 **AGENT**: Bunları doğrula:

```bash
gh auth status                                    # ✓ RsGoksel logged in
git config --get user.email                       # ✓ gokssel.gunduz@gmail.com
ls "<project>/docs/paper.tex"                     # ✓ LaTeX kaynağı var mı
ls "<project>/docs/paper_draft.md"                # ✓ markdown taslak var mı
ls "<project>/results/plots/"                     # ✓ figürler hazır mı
```

Yoksa ya kullanıcıdan iste ya da paper'ın bulunduğu yerden topla.

---

## ADIM 1 — Secret Taraması (KRİTİK, ATLAMA)

🤖 **AGENT**: Tüm proje dizininde secret tara:

```bash
# IBM Quantum token paterni: 32-char alphanumeric
grep -rE "(IBM_QUANTUM_TOKEN|api_key|API_KEY|gho_[a-zA-Z0-9]{20,}|sk-[a-zA-Z0-9]{20,}|token\s*=\s*['\"][a-zA-Z0-9_]{20,})" <project>/

# Bilinen IBM token formatı (32 karakter, harfler+sayılar)
grep -rE "token\s*=\s*['\"][a-zA-Z0-9]{32,}['\"]" <project>/
```

Eğer gerçek token bulunursa:
1. **Kullanıcıyı uyar:** "Token X dosyasında satır Y'de buldum. GitHub'a gitmeden değiştiriyorum, ama yine de IBM panelinden revoke et: https://quantum.ibm.com/account"
2. Token yerine `'YOUR_IBM_QUANTUM_TOKEN'` placeholder yaz
3. Sanitization'ı `git init`'ten **önce** yap, yoksa git history'de kalır

🧑 **USER**: Token revoke etmen gerekirse hatırlatma yap.

---

## ADIM 2 — GitHub Repo Hazırlığı

🤖 **AGENT**: Bu dosyaları repo köküne yaz:

### 2.1 `README.md` Şablonu

```markdown
# <Project Title>

[![DOI](https://img.shields.io/badge/DOI-10.XXXXX%2F...-blue.svg)](https://doi.org/<DOI>)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper License: CC BY 4.0](https://img.shields.io/badge/Paper-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)

<One-paragraph plain-English description.>

> 📄 **Preprint:** Gündüz, K. G. (YEAR). *<Title>*. ResearchGate. [https://doi.org/<DOI>](https://doi.org/<DOI>)

## Highlights
## Pipeline (ASCII diagram)
## Headline Results (markdown tables)
## Repository Layout (tree)
## Setup
## Reproducing Paper Results
## Limitations
## Citation (bibtex with DOI)
## License (MIT for code, CC BY 4.0 for paper)
## Acknowledgements
```

DOI başta yoksa placeholder bırak; ADIM 7'de doldurulacak.

### 2.2 `LICENSE`

MIT License, copyright satırı: `Copyright (c) <YEAR> Kadir Göksel Gündüz`

### 2.3 `.gitignore`

Python + venv + IDE + secrets:
```
__pycache__/
*.py[cod]
.venv/
venv/
.env
.env.local
.qiskit/
*.pem
.vscode/
.idea/
.DS_Store
Thumbs.db
.ipynb_checkpoints/
.pytest_cache/
.mypy_cache/
*.log
scratch/
tmp/
```

### 2.4 `requirements.txt`

Pinned versiyonlar — paper'ın Appendix B'sindekiyle aynı olmalı. Standart stack:
```
torch==2.6.0
numpy>=1.26
transformers==5.3.0
datasets==4.6.1
scikit-learn==1.7.2
qiskit==2.3.0
qiskit-ibm-runtime==0.45.1
matplotlib==3.10.7
seaborn==0.13.2
```

### 2.5 `CITATION.cff`

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite the accompanying paper."
type: software
title: "<Project Title>"
authors:
  - family-names: "Gündüz"
    given-names: "Kadir Göksel"
    alias: "RsGoksel"
    email: "gokssel.gunduz@gmail.com"
    orcid: "https://orcid.org/0009-0007-9120-5659"
    affiliation: "Energy Institute, Istanbul Technical University (İTÜ)"
repository-code: "https://github.com/RsGoksel/<repo-name>"
url: "https://doi.org/<DOI-after-step-7>"
identifiers:
  - type: doi
    value: "<DOI-after-step-7>"
    description: "ResearchGate preprint DOI"
license: MIT
keywords: [<5-10 keywords>]
preferred-citation:
  type: generic
  title: "<Paper Title>"
  year: <YEAR>
  month: <MONTH>
  authors:
    - family-names: "Gündüz"
      given-names: "Kadir Göksel"
      orcid: "https://orcid.org/0009-0007-9120-5659"
      affiliation: "Energy Institute, Istanbul Technical University (İTÜ)"
  doi: "<DOI-after-step-7>"
  publisher:
    name: "ResearchGate"
  notes: "Preprint"
```

---

## ADIM 3 — Paper Header Güncelleme

🤖 **AGENT**: Hem `paper_draft.md` hem `paper.tex`'te author block'u şu hale getir:

### Markdown:
```markdown
**Author:** Kadir Göksel Gündüz¹*
**Affiliation:** ¹ Energy Institute, Istanbul Technical University (İTÜ), Istanbul, Türkiye
**Email:** gokssel.gunduz@gmail.com
**ORCID:** [0009-0007-9120-5659](https://orcid.org/0009-0007-9120-5659)
**Date:** <DD Month YEAR>
**DOI:** [<DOI-after-step-7>](https://doi.org/<DOI>)
**Preprint:** ResearchGate · License: CC BY 4.0
**Code & Data:** [github.com/RsGoksel/<repo-name>](https://github.com/RsGoksel/<repo-name>)

*Corresponding author.
```

### LaTeX (IEEE/MDPI):
```latex
\usepackage{orcidlink}
\author{\IEEEauthorblockN{Kadir G\"{o}ksel G\"{u}nd\"{u}z\,\orcidlink{0009-0007-9120-5659}}
\IEEEauthorblockA{Energy Institute\\
Istanbul Technical University (\.{I}T\"{U})\\
Istanbul, T\"{u}rkiye\\
Email: \texttt{gokssel.gunduz@gmail.com}\\
ORCID: \href{https://orcid.org/0009-0007-9120-5659}{0009-0007-9120-5659}\\
DOI: \href{https://doi.org/<DOI>}{<DOI>}}}
```

### Code & Data Availability + Declarations (References'tan önce)

```markdown
## Code and Data Availability
The full source code, trained checkpoints, cached embeddings, IBM Quantum job logs,
and all paper figures are publicly available at https://github.com/RsGoksel/<repo>
under the MIT License. The repository includes scripts to fully reproduce every
result reported in this paper.

## Declarations
**Conflict of Interest.** The author declares no competing interests.
**Funding.** This research received no specific grant from any public, commercial,
or not-for-profit funding agency.
**IBM Quantum Access.** Hardware experiments were conducted under the IBM Quantum
Open Plan (10 min/month free tier).
```

---

## ADIM 4 — Git Init + Push

🤖 **AGENT**:

```bash
cd <project>
git init -b main
git add .

# Son secret kontrolü (zaten staged dosyalarda)
git diff --cached --name-only | xargs grep -lE "([a-zA-Z0-9]{32,}|gho_[a-zA-Z0-9]+)" 2>/dev/null

# Boyut kontrolü
du -sh .  # 100MB üstündeyse ki büyük .pt dosyalarını gitignore'a ekle

git commit -m "$(cat <<'EOF'
Initial commit: <one-line description>

<2-3 paragraf açıklama: ne yapıldı, ne içerir, neden public>
EOF
)"

# Repo oluştur ve push (gh CLI tek komutta yapar)
gh repo create <repo-name> \
  --public \
  --source=. \
  --remote=origin \
  --description "<140 char description>" \
  --push

# Topics ekle (max 20)
gh repo edit RsGoksel/<repo-name> \
  --add-topic <topic1> --add-topic <topic2> ... \
  --homepage "https://doi.org/<DOI>"   # DOI sonradan eklenir
```

🧑 **USER**: gh CLI yetkisi yoksa "gh auth login" yap; çoğunlukla zaten kayıtlı.

---

## ADIM 5 — PDF Üret (Overleaf)

🤖 **AGENT**: Overleaf'in derleyebileceği flat ZIP hazırla:

```bash
mkdir -p /tmp/overleaf_pkg
cp <project>/docs/paper.tex /tmp/overleaf_pkg/
# Tüm \includegraphics dosyalarını grep ile bul ve kopyala
grep -oE '\\includegraphics[^{]*\{[^}]+\}' <project>/docs/paper.tex \
  | sed 's/.*{\(.*\)}/\1/' \
  | xargs -I{} find <project> -name "{}" -exec cp {} /tmp/overleaf_pkg/ \;

cd /tmp/overleaf_pkg && zip -r "<user-desktop>/<repo-name>-overleaf.zip" .
```

🧑 **USER**:
1. https://www.overleaf.com → New Project → Upload Project → ZIP'i sürükle
2. Recompile (yeşil buton)
3. Hata yoksa → Download PDF
4. Hata varsa: agent'a hata mesajını gönder, paper.tex'i düzelt

**Yaygın Overleaf hataları ve çözümleri:**
| Hata | Sebep | Çözüm |
|------|-------|-------|
| `Missing package: orcidlink` | Eski TeX Live | `Settings → TeX Live version → 2023+` |
| `File not found: figX.png` | Figür yolu yanlış | ZIP'e flat hale getir, `\graphicspath{{./}}` kullan |
| `Unicode character` | UTF-8 sorun | `\usepackage[utf8]{inputenc}` ekle |
| `Türkiye` italik bozuk | Karakter encoding | LaTeX'te `T\"{u}rkiye` kullan |

---

## ADIM 6 — ResearchGate Upload (Manuel + Otomasyon Talimatı)

🧑 **USER** (agent yardımıyla): https://www.researchgate.net → Add new → Publication

| Alan | Değer | Not |
|------|-------|-----|
| **Publication type** | `Preprint` | Konferansta sunulmadıkça asla "Conference Paper" seçme |
| **File** | İndirilen PDF | Sürükle-bırak |
| **Upload type** | `Add only a public file` | Yedek için private gerek yok |
| **Title** | Tam başlık (boşluklu, alt çizgi yok) | Dosya adıyla karıştırma |
| **Authors** | Otomatik (ResearchGate profili) | Co-author varsa ekle |
| **Date** | Paper'daki tarih | DD/Month/YYYY |
| **DOI** | **BOŞ BIRAK** | ResearchGate sonradan üretiyor |
| **Description** | Paper'ın Abstract'ı | Birebir kopyala |
| **Has this been peer-reviewed?** | **`No, it hasn't been peer reviewed`** | KRİTİK: "Yes" akademik dürüstlük ihlali |
| **Licence** | **`CC BY 4.0`** | Akademik standart, maksimum atıf |

🧑 **USER**: Submit'e bas. Sayfa ilerleyince:

> **"Want to generate a DOI for your preprint?"** — **YES, GENERATE DOI**

DOI üretimi 1-5 dakika sürer. Format: `10.13140/RG.2.2.XXXXX.XXXXX`

---

## ADIM 7 — Post-Publication: DOI'yi Her Yere Ekle

🧑 **USER**: DOI'yi agent'a ver: "DOI: 10.13140/RG.2.2.XXXXX.XXXXX"

🤖 **AGENT**:

### 7.1 README.md
- Üst satıra DOI badge ekle: `[![DOI](https://img.shields.io/badge/DOI-<encoded>-blue.svg)](https://doi.org/<DOI>)`
- "Preprint:" satırına DOI link
- Bibtex'e `doi = {<DOI>}` ve `url = {https://doi.org/<DOI>}` alanlarını ekle

### 7.2 CITATION.cff
```yaml
url: "https://doi.org/<DOI>"
identifiers:
  - type: doi
    value: "<DOI>"
    description: "ResearchGate preprint DOI"
preferred-citation:
  doi: "<DOI>"
  url: "https://doi.org/<DOI>"
  publisher:
    name: "ResearchGate"
```

### 7.3 paper_draft.md ve paper.tex
Author block'a DOI satırı ekle (şablon ADIM 3'te).

### 7.4 GitHub repo metadata
```bash
gh repo edit RsGoksel/<repo> --homepage "https://doi.org/<DOI>"
```

### 7.5 Commit + push
```bash
git add README.md CITATION.cff docs/paper_draft.md docs/paper.tex
git commit -m "Add DOI <DOI> across all metadata

Preprint published on ResearchGate (<MONTH YEAR>, CC BY 4.0):
- README: DOI badges + bibtex
- CITATION.cff: DOI in identifiers and preferred-citation
- paper_draft.md + paper.tex: DOI in author block"
git push
```

---

## ADIM 8 — Yayın Sonrası Görünürlük (Opsiyonel ama Önerilen)

🧑 **USER**:

1. **IBM Quantum token revoke** (eğer ADIM 1'de bulunduysa) — https://quantum.ibm.com/account
2. **arXiv'e yükle** — endorsement gerekiyor; tanıdık akademisyenden iste. cs.LG (primary) + quant-ph (cross-list).
   - Comment alanı: `Preprint, also available at ResearchGate. DOI: <DOI>`
3. **LinkedIn / Twitter / Bluesky duyurusu** — ilk 24 saat algoritmik öneri için kritik
4. **ITÜ Enerji Enstitüsü intranet/mailing list** — iç atıf zinciri
5. **Google Scholar profilinde manual ekleme** — Scholar otomatik bulmazsa

---

## SIK YAPILAN HATALAR (Tekrar etme!)

| ❌ Hata | ✅ Doğrusu |
|---------|------------|
| "Conference Paper" seçmek | Preprint, peer-reviewed olmadıkça |
| "Yes, peer-reviewed" işaretlemek | NO — akademik dürüstlük |
| CC BY-NC veya ND seçmek | CC BY 4.0 (en permissive) |
| Token'ı dosyada bırakıp commit | ADIM 1 secret tarama, sonra git init |
| `git push --force` | Asla. Yeni commit at. |
| DOI'yi ResearchGate dışında üretmeye çalışmak | ResearchGate kendi üretiyor, "Generate DOI" butonu |
| Title alanına dosya adı yapıştırmak | Boşluklu, normal İngilizce başlık |
| Code repo'sunu private bırakmak | Public — atıf değeri için |
| `__pycache__` push etmek | .gitignore'a ekle, başta sil |
| Affiliation: "Independent Researcher" yazmak | İTÜ Energy Institute'a bağlısın, yaz |

---

## ÇALIŞTIRILABİLİR KISA-YOLLAR (Auto Mode için)

```bash
# Tam akış (kullanıcı "yayınla" dediğinde)
1. Secret scan        → grep + sanitize
2. Repo files yaz     → README, LICENSE, .gitignore, requirements.txt, CITATION.cff
3. Author block       → paper_draft.md + paper.tex
4. Git init + push    → gh repo create --public --push
5. ZIP for Overleaf   → /Masaüstü/<repo>-overleaf.zip
6. (USER) Overleaf → PDF → ResearchGate → "Generate DOI" → bekle
7. DOI'yi al, dağıt   → README + CITATION + paper.* + gh repo edit
8. Final push         → tek commit, "Add DOI X across all metadata"
```

---

## Geçmiş Yayınlar (Referans)

| Tarih | Repo | DOI | Konu |
|-------|------|-----|------|
| 2026-04-30 | [quantum-encoding-nlp](https://github.com/RsGoksel/quantum-encoding-nlp) | [10.13140/RG.2.2.32211.13607](https://doi.org/10.13140/RG.2.2.32211.13607) | VQC encoding selection for NLP sentiment analysis |
| `<sıradaki>` | `<sıradaki-repo>` | `<sıradaki-DOI>` | `<sıradaki-konu>` |

Yeni yayın eklendiğinde bu tabloya satır ekle.
