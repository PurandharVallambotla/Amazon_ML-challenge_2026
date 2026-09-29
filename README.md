# Amazon ML Challenge 2026: Business Entity Resolution

## Overview
This repository contains my solution for the Amazon ML Challenge 2026. The objective of the challenge was to perform Entity Resolution—identifying and matching business entities across multiple disparate data sources using varied schema and noisy text fields (e.g., names, addresses, localities).

*Note: For compliance and data privacy, the raw datasets (`dataset/`), output files (`output/`), and final submission zips have been explicitly excluded from this repository.*

## Approach & Methodology

My approach evolved from naive string matching to a robust, scalable, machine-learning-driven pipeline leveraging inverted indexing for blocking and XGBoost for classification.

### 1. Blocking Strategy (Candidate Generation)
Comparing every entity against every other entity is computationally infeasible ($O(N^2)$). To scale the solution, I implemented a custom **TF-IDF weighted Inverted Index**:
- **Composite Keys**: Extracted normalized full names, significant words (ignoring generic industry terms), core bigrams, and alphanumeric address/postal code combinations.
- **Scoring**: Instead of strict boolean matches, candidates were retrieved based on inverse document frequency (IDF). Rare words (e.g., specific brand names) were given higher weight than common ones, dramatically reducing the candidate search space while maximizing recall.

### 2. Feature Engineering
For every candidate pair retrieved by the blocking step, I generated 27 discriminative, country-agnostic features:
- **Lexical/Fuzzy Metrics**: Token sort ratio, partial ratio, and set ratio on the "core" business name.
- **Brand Logic Extraction**: Developed custom functions to strip out generic industry suffixes and identify the distinguishing "brand" of a business, matching those explicitly.
- **Geographic & Number Compatibility**: Parsed primary house numbers, postal codes, and geographic tokens. I heavily penalized pairs with conflicting primary address numbers while rewarding partial subset matches (e.g., "Unit A" vs. "Unit A & B").
- **High-Precision Heuristics**: Added specific boolean flags (e.g., `addr_match_brand_conf`, `perfect_align`) to capture critical edge cases where models often fail.

### 3. Classification
A tuned **XGBoost Classifier** (300 estimators, max depth 6) was trained on the engineered features. XGBoost was chosen for its excellent handling of non-linear interactions among tabular features, fast training, and robustness to outliers in fuzzy similarity scores.

## Chain of Thought & Evolution

1. **Initial Naive Approach**: I started with simple Levenshtein distance across full names. *Result*: Exceptionally slow and high false positive rate due to common legal suffixes ("LLC", "Inc").
2. **Implementing Blocking**: Shifted to an inverted index based on exact word matches. *Result*: Improved speed, but missed many true positives (high False Negative rate) due to misspellings or missing address data.
3. **Advanced Blocking & Brand Logic**: Incorporated TF-IDF so that matching a unique identifier like a postal code or rare brand name pulled the candidate, while generic words didn't overwhelm the index. I also noticed through error analysis (e.g., via diagnostic scripts like `analyze_india_missed.py` and `analyze_lost_tps.py`) that region-specific formatting was breaking the matcher.
4. **Final Model**: I aggregated the similarities and discrepancies into a vector and let XGBoost learn the importance of each feature.

## Lessons Learned & Mistakes Made
Throughout the development process, I encountered several pitfalls:
- **Aggressive Early Blocking**: Initially, my blocking criteria were too strict, causing me to lose True Positives before the model even saw them. The lesson was to optimize blocking for *Recall*, leaving *Precision* to the XGBoost model.
- **Generic Word Collisions**: I didn't initially filter out generic industry words (e.g., "Consulting", "Services"). This polluted the inverted index and caused massive slowdowns and false positives.
- **Address Over-reliance**: I originally treated address differences as strong negative signals. However, exploratory data analysis showed that many branches of the same entity have conflicting numbers (e.g., "123 Main St" vs "125 Main St"). Hard penalization led to false negatives.

## Future Alternatives to Explore
If I had more time or computational resources, I would explore:
1. **Semantic Deep Learning**: Instead of lexical fuzzy matching, use fine-tuned Transformer models (like **Sentence-BERT** or **RoBERTa**) to generate dense embeddings for names and addresses, capturing semantic similarity.
2. **Graph-Based Entity Resolution**: Framing the problem as a graph where nodes are entities and edges are similarities. Graph Neural Networks (GNNs) could propagate information to find transitive matches (e.g., A matches B, B matches C, infer A matches C).
3. **Active Learning**: Instead of relying purely on the static ground truth, use an Active Learning framework to manually label edge-case pairs where the XGBoost model was most uncertain, rapidly improving performance in difficult domains.

---

I am proud of the robustness of the feature engineering and the scalability of the custom indexing solution. Feel free to explore the `code/` and `scratch/` directories to view the specific implementation details and the diagnostic scripts used to iteratively improve the pipeline.
