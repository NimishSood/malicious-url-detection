# Malicious URL Detection

This repository is a statistics/data mining course project on malicious URL detection. It now has two complementary layers:

- [`notebooks/file1.ipynb`](notebooks/file1.ipynb): data acquisition, cleaning, EDA, and descriptive statistical reasoning
- [`notebooks/02_statistical_modeling.ipynb`](notebooks/02_statistical_modeling.ipynb): supervised modelling, model comparison, test evaluation, and interpretation

The project is designed to demonstrate four course skills:

1. Statistical Modelling
2. Statistical Reasoning and Interpretation
3. Analytical Question Formulation
4. Reproducibility and Professional Practice

## Analytical Questions

The modelling layer is organized around four questions:

1. Which lexical URL features are most associated with maliciousness?
2. How well can lexical features distinguish benign from malicious URLs?
3. Which malicious subtype is hardest to classify?
4. Do suspicious-token features add useful signal beyond the core lexical counts?

## Dataset

- Source: [Kaggle - Malicious URLs Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)
- Local file: `data/malicious_phish.csv`
- Raw schema:
  - `url`: raw URL text
  - `type`: label (`benign`, `phishing`, `malware`, `defacement`)

For the current checked-in dataset snapshot, the reproducible cleaning pipeline keeps `640,825` rows and removes malformed, duplicated, or conflicting URL records. Current cleaned class counts are:

- `benign`: 427,932
- `defacement`: 95,308
- `phishing`: 93,940
- `malware`: 23,645

The exact file hash used in the saved results is recorded in [`results/run_metadata.json`](results/run_metadata.json).

## Workflow

### 1. Reproducible Data Loading and Cleaning

The project uses explicit, deterministic preprocessing:

1. Normalize string formatting and labels
2. Drop missing URL or label values
3. Drop empty URL or label values after stripping
4. Keep only the four expected labels
5. Remove URLs containing whitespace
6. Remove URLs containing control characters
7. Remove exact duplicate rows
8. Remove duplicate URLs with conflicting labels

Cleaning audit tables are saved in [`results/cleaning_audit.csv`](results/cleaning_audit.csv).

### 2. Feature Engineering

The original notebook engineers lexical features for EDA. The modelling code reuses and extends that work with interpretable features such as:

- URL length and character counts
- host length, path length, query length
- path depth and subdomain count
- indicators for query strings, schemes, IP-based hosts, `@`, and percent encoding
- suspicious token indicators: `login`, `verify`, `account`, `secure`, `update`, `bank`

These are implemented in [`src/features.py`](src/features.py).

### 3. Supervised Modelling Tasks

Two classification tasks are included:

1. Binary classification: `benign` vs `malicious`
2. Multiclass classification: `benign`, `phishing`, `malware`, `defacement`

The modelling notebook keeps the workflow course-aligned:

- stratified train/test split
- cross-validation on the training split only
- model tuning before final test evaluation
- confusion matrices and per-class test metrics
- interpretation after each major result section

### 4. Models Compared

Three model families are compared for each task:

- Majority-class baseline (`DummyClassifier`)
- Logistic Regression
- k-Nearest Neighbours

Logistic regression is the main interpretive model. k-NN is included because it aligns with the lecture material and provides a nonlinear comparison point.

### 5. Evaluation Strategy

Evaluation emphasizes class imbalance and out-of-sample performance:

- fixed random seed: `42`
- stratified test split
- `3`-fold cross-validation for tuning
- accuracy
- balanced accuracy
- precision / recall / F1
- macro F1
- per-class metrics
- confusion matrices

Saved result tables include:

- [`results/binary_feature_set_comparison.csv`](results/binary_feature_set_comparison.csv)
- [`results/binary_cv_results.csv`](results/binary_cv_results.csv)
- [`results/binary_test_metrics.csv`](results/binary_test_metrics.csv)
- [`results/multiclass_cv_results.csv`](results/multiclass_cv_results.csv)
- [`results/multiclass_test_metrics.csv`](results/multiclass_test_metrics.csv)

### 6. Reproducibility Choice for k-NN

The cleaned dataset is large enough that repeated cross-validation for k-NN on the full dataset would be unnecessarily expensive for a course project. To keep the workflow runnable on ordinary hardware, the modelling pipeline draws a reproducible stratified sample of `80,000` cleaned URLs for the supervised section. That choice is explicit in the notebook, code, and saved metadata.

## Current Interpretation Summary

The current saved results suggest:

- k-NN is the strongest predictive model on both tasks
- logistic regression is less accurate, but much easier to interpret
- phishing is the hardest class to separate in the multiclass setting
- suspicious-token features add only a small validation gain beyond the core lexical feature set

These are associations in this dataset, not causal claims.

## Repository Structure

```text
malicious-url-detection
├── data
│   └── malicious_phish.csv
├── notebooks
│   ├── file1.ipynb
│   └── 02_statistical_modeling.ipynb
├── results
│   ├── *.csv
│   └── run_metadata.json
├── src
│   ├── __init__.py
│   ├── evaluate.py
│   ├── features.py
│   └── train.py
├── requirements.txt
├── README.md
└── MaliciousURL_Analysis_Improved.pptx
```

## How to Run

1. Create and activate a virtual environment with Python 3.11 or newer.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Generate the modelling outputs:

   ```bash
   python -m src.train
   ```

4. Start JupyterLab:

   ```bash
   jupyter lab
   ```

5. Run the notebooks in order if you want the full report flow:

   - `notebooks/file1.ipynb`
   - `notebooks/02_statistical_modeling.ipynb`

## Limitations

- The models are lexical-only and do not use page content, DNS, WHOIS, or temporal context.
- k-NN predicts well here, but it is less interpretable and more computationally expensive than logistic regression.
- Logistic regression coefficients describe association, not proof of cause.
- The dataset is imbalanced, so headline accuracy alone is not reliable.
- Results depend on the quality and representativeness of the labels in the source dataset.
