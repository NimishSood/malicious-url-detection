# Malicious URL Detection

This project analyzes a labeled URL dataset to demonstrate five core data science competencies:

1. Data Acquisition
2. Data Understanding
3. Data Cleaning
4. Exploratory Data Analysis (EDA)
5. Statistical Reasoning and Interpretation

The main analytical report is in [`notebooks/file1.ipynb`](notebooks/file1.ipynb).

## Project Motivation

Cyberattacks frequently rely on deceptive URLs to steal credentials, distribute malware, or direct users to compromised pages. This project explores whether structural URL patterns can support malicious URL analysis.

## Dataset

- Source: [Kaggle - Malicious URLs Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)
- Local file: `data/malicious_phish.csv`
- Raw shape: 651,191 rows x 2 columns
- Columns:
  - `url`: raw URL text
  - `type`: label (`benign`, `phishing`, `malware`, `defacement`)

The notebook also documents the upstream source collections referenced by the dataset creator (ISCX-URL-2016, Malware Domain Blacklist, Faizan repository, PhishTank, PhishStorm).

## Notebook Workflow

### 1) Data Acquisition

- Documents dataset origin and class-label context.
- Uses reproducible relative-path loading logic.
- Validates required columns (`url`, `type`).
- Captures local data provenance (file path, size, SHA-256 hash).

### 2) Data Understanding

- Provides a data dictionary and class definitions.
- Reports shape, dtypes, sample records, and class balance.
- Summarizes URL-length behavior and explicit scheme usage by class.

### 3) Data Cleaning

Implements a deterministic, auditable cleaning pipeline:

1. Standardize string formatting (`strip`, lowercase labels)
2. Remove null/empty URL or label rows
3. Keep only expected labels
4. Remove URLs with whitespace/control characters
5. Remove exact duplicate rows
6. Remove exact duplicate URLs with conflicting labels

The notebook outputs a step-by-step cleaning audit table showing rows removed per rule.

### 4) Exploratory Data Analysis (EDA)

EDA includes:

- Class distribution with imbalance interpretation
- URL length distribution and per-class comparison
- Lexical feature analysis (`url_length`, `digit_count`, `dot_count`, `hyphen_count`, `slash_count`)
- Correlation heatmap of lexical features
- Suspicious token prevalence by class (`login`, `verify`, `account`, `secure`, `update`, `bank`)
- TLD composition by class

Each visualization is paired with interpretation text.

### 5) Statistical Reasoning and Interpretation

The notebook now explicitly demonstrates statistical reasoning through:

- `4.2 URL length behavior`: distribution shape and class-level spread, with interpretation of skew and differing medians
- `4.3 Lexical feature patterns`: class-level averages and correlation structure for core lexical features
- `4.4 Statistical reasoning and uncertainty`: descriptive summaries by class (`mean`, `median`, `std`, `IQR`, `95% CI`, coefficient of variation) plus uncertainty-aware interpretation
- `4.4 Statistical reasoning and uncertainty`: Spearman correlation ranking and Wilson confidence intervals for query-string prevalence and IP-host usage
- `4.5` and `4.6`: Markdown interpretation that explains what token-prevalence and TLD-composition outputs show, why they matter, and what limitations remain

What was added:

- A dedicated statistical reasoning subsection in the notebook
- Explicit discussion of variability, spread, and uncertainty
- Clear Markdown explanations for the main statistical outputs and their implications

## Repository Structure

```text
malicious-url-detection
├── data
│   └── malicious_phish.csv
├── notebooks
│   └── file1.ipynb
├── src
│   └── feature_engineering.py
├── requirements.txt
└── README.md
```

## How to Run

1. Create and activate a virtual environment (Python 3.11+ recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start Jupyter:
   ```bash
   jupyter lab
   ```
4. Open `notebooks/file1.ipynb` and run all cells in order.

## Current Scope and Next Steps

Current notebook scope is data acquisition, understanding, cleaning, EDA, and statistical reasoning/interpretation. Model training/evaluation can be added next (for example logistic regression, tree-based models, and class-aware evaluation metrics).
