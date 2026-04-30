# Design Doc: Gradient Trainability as Encoding Selection Criterion

**Date:** 2026-03-08
**Status:** APPROVED
**Type:** Short paper (4-6 pages)

---

## Research Question

"Can gradient variance serve as a reliable pre-training diagnostic for selecting quantum data encoding strategies in NLP sentiment analysis tasks?"

## Core Contribution

A systematic empirical study showing that gradient trainability (measured via parameter gradient variance) correlates with downstream NLP task accuracy across quantum encoding strategies, and that encoding-specific preprocessing (sigmoid scaling) is a critical but undocumented factor.

## Experimental Design

### Phase 7A: Multi-Seed Validation
- Seeds: [42, 123, 456, 789, 2024]
- Models: 4 encodings (angle, dense, iqp, reupload)
- Dataset: IMDb (2000 subset)
- Output: mean accuracy + std for each encoding
- Purpose: Statistical reliability of single-seed results

### Phase 7B: SST-2 Generalization
- Seeds: [42]
- Models: 4 encodings + baseline + attention
- Dataset: SST-2 (2000 subset)
- Output: Cross-dataset encoding ranking comparison
- Purpose: Does angle > dense > IQP > reupload hold on second dataset?

### Phase 7C: Visualization & Figures
1. Gradient variance vs accuracy scatter plot (both datasets)
2. Training curves (all encodings, same axes)
3. Qubit/depth scaling gradient variance plots
4. Circuit diagrams for 4 encodings

### Phase 7D: Paper Writing
- LaTeX or markdown draft
- Abstract, Introduction, Methods, Results, Discussion, Conclusion

## Success Criteria

1. Multi-seed confirms encoding ranking (angle best, IQP/reupload worst)
2. SST-2 shows same or similar ranking pattern
3. Gradient variance-accuracy correlation is visually and statistically clear
4. Paper honestly reports limitations without overclaiming

## Risks

| Risk | Mitigation |
|------|-----------|
| Multi-seed reverses ranking | Report honestly, analyze variance |
| SST-2 shows different ranking | Discuss dataset-specific effects |
| Correlation is weak | Report as finding, add more encodings |
| Training too slow (20 experiments) | Parallelize carefully (max 3 at once) |
