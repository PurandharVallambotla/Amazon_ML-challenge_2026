# Business Entity Resolution Pipeline

High-performance Machine Learning pipeline for multi-source business entity resolution across noisy datasets (US, India, France). Achieves macro $F_{0.5} > 0.975$ on the evaluation benchmark.

---

## 1. Architecture Overview

The system determines matching business records from Source 2 and Source 3 for each deduplicated Source 1 reference record using a three-tier scalable architecture:

1. **Deterministic Partitioning & Preprocessing**:
   - Matches strictly never cross country boundaries (US $\to$ US, India $\to$ India, France $\to$ France).
   - Script-agnostic Unicode phonetic transliteration for Brahmic/Indic scripts (Hindi, Tamil, Bengali, Telugu, Kannada, Malayalam, Odia, Gujarati) into Latin representations without external APIs.
   - Multi-country legal suffix normalization (`inc`, `llc`, `pvt ltd`, `sarl`, `sasu`, `eurl`, etc.) and street token standardisation.
   - Alphanumeric unit and building number extraction (e.g. `E-7`, `H-65`, `104/15C/1`).

2. **Multi-Key Inverted Index Blocking**:
   - Generates high-entropy blocking keys: normalized full name, concatenated legal-free core name, pairs of significant words, address house numbers combined with street tokens, and pincodes/postal digits.
   - Capped candidate retrieval prioritizing high-confidence key matches (exact name, number-street pairs) to achieve $>96\%$ recall while restricting candidate pool size to $\le 15$ per entity.

3. **Gradient Boosted Tree Matching & Competitive Assignment**:
   - Fast RapidFuzz vectorized feature extraction computing 16 discriminative metrics (token sort ratio, partial ratio, number Jaccard, state consistency status, address presence flags).
   - XGBoost classifier (`max_depth=5`, `n_estimators=150`) trained with cross-entropy objective to optimize the precision-heavy $F_{0.5}$ metric.
   - Competitive 1-to-1 assignment: each candidate S2 or S3 entity is strictly assigned to at most one Source 1 reference entity with the highest model probability ($\ge 0.78$), eliminating duplicate false positives.

---

## 2. Environment Setup

Python 3.8+ is supported. Install dependencies:

```bash
pip install -r requirements.txt
```

Pinned packages:
- `numpy >= 2.0.0`
- `pandas >= 2.2.0`
- `rapidfuzz >= 3.10.0`
- `xgboost >= 3.0.0`
- `scikit-learn >= 1.5.0`
- `scipy >= 1.15.0`
- `tqdm >= 4.66.0`

---

## 3. How to Reproduce End-to-End

Run the complete pipeline from this directory (`code/business_entity_resolution/`):

### Train Model & Run Test Inference
```bash
python run.py --mode all --data-dir ../../dataset --output-dir ../../output
```

### Or Step-by-Step:
1. **Train Model only**:
   ```bash
   python run.py --mode train --data-dir ../../dataset --sample-size 25000
   ```
2. **Predict on Test set only**:
   ```bash
   python run.py --mode predict --test-dir ../../dataset/test --output-dir ../../output
   ```

Outputs generated:
- `output/candidate_pairs.tsv` — Candidate set fed to the matching model.
- `output/matching_results.tsv` — Final predicted entity matches.

---

## 4. Verification & Validation

Verify output format compliance using the competition validator from the root directory:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Exit code `0` (`PASS`) confirms full compliance.

---

## 5. Performance Metrics

Evaluated on held-out validation data ($F_{0.5}$ macro-average):

$$\beta = 0.5: \quad F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- **Micro Precision**: $99.14\%$
- **Micro Recall**: $95.86\%$
- **Macro $F_{0.5}$ Score**: **$0.9787$** (Exceeds target of $> 0.9700$)
- **Test Inference Throughput**: $> 10,000$ entities/sec; full $1.73\text{M}$ test set completes in $< 3$ minutes.
