# Malicious URL Detection

This repository contains an end-to-end exploratory analysis of a labeled malicious URL dataset in [`notebooks/file1.ipynb`](notebooks/file1.ipynb). The notebook is structured as a reproducible analytical report covering data acquisition, data understanding, data cleaning, exploratory data analysis (EDA), and statistical reasoning.

## Project Focus

The notebook is designed to demonstrate these core competencies:

1. Reproducible data loading and provenance tracking
2. Dataset understanding and class-balance analysis
3. Deterministic, auditable data cleaning
4. Exploratory analysis of lexical URL patterns
5. Statistical reasoning with spread and uncertainty measures

## Dataset

- Source: [Kaggle - Malicious URLs Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)
- Local file: `data/malicious_phish.csv`
- Raw shape: 651,191 rows x 2 columns
- Columns:
  - `url`: raw URL text
  - `type`: label (`benign`, `phishing`, `malware`, `defacement`)

The notebook also documents the upstream collections referenced by the dataset creator, including ISCX-URL-2016, Malware Domain Blacklist, Faizan repository, PhishTank, and PhishStorm.

## Notebook Workflow

### 1. Reproducibility and Data Acquisition

- Uses relative-path loading so the notebook runs from either the repository root or `notebooks/`
- Validates the expected schema and keeps fallback handling explicit
- Prints local file provenance:
  - resolved path
  - file size
  - SHA-256 hash
  - row count
  - loaded columns

### 2. Data Understanding

- Defines the dataset schema and label meanings
- Reports shape, data types, sample rows, and class balance
- Summarizes URL length by class
- Measures explicit `http`/`https` scheme usage by class

Current raw class balance from the notebook:

- `benign`: 428,103
- `defacement`: 96,457
- `phishing`: 94,111
- `malware`: 32,520

### 3. Data Cleaning

The notebook applies a deterministic cleaning pipeline:

1. Normalize string formatting and labels
2. Drop missing URL or label values
3. Drop empty URL or label values after stripping
4. Keep only expected labels
5. Remove URLs containing whitespace
6. Remove URLs containing control characters
7. Remove exact duplicate rows
8. Remove duplicate URLs with conflicting labels

Cleaning audit reported in the notebook:

- Rows before cleaning: 651,191
- Rows after cleaning: 640,792
- Rows removed total: 10,399

Post-cleaning class counts:

- `benign`: 427,931
- `defacement`: 95,285
- `phishing`: 93,931
- `malware`: 23,645

### 4. Exploratory Data Analysis

EDA in the notebook includes:

- Class distribution plots for the cleaned dataset
- URL length distribution and class-wise comparison
- Lexical feature engineering:
  - `url_length`
  - `digit_count`
  - `dot_count`
  - `hyphen_count`
  - `slash_count`
  - `has_query`
  - `has_scheme`
  - `has_ip_host`
  - `tld`
- Mean feature heatmaps by class
- Correlation analysis for core lexical features
- Suspicious token prevalence by class (`login`, `verify`, `account`, `secure`, `update`, `bank`)
- Top-TLD composition by class

### 5. Statistical Reasoning and Interpretation

The notebook now makes the statistical reasoning explicit rather than leaving it implied in the charts.

It includes:

- Descriptive summaries by class for URL length:
  - count
  - mean
  - median
  - standard deviation
  - quartiles
  - IQR
  - 95th percentile
  - coefficient of variation
  - mean-minus-median gap
  - 95% confidence intervals for the mean
- Spearman correlation ranking for lexical features
- Wilson confidence intervals for:
  - query-string prevalence
  - IP-host prevalence
- Interpretation text after each major section to explain what the patterns mean and what their limits are

Examples of current notebook findings:

- `defacement` URLs have the highest typical length
- `phishing` URLs show the highest relative variability in length
- `url_length` and `slash_count` have the strongest monotonic relationship among the core lexical features
- `defacement` URLs show the highest query-string rate
- `malware` URLs stand out for IP-based hosts

## Repository Structure

```text
malicious-url-detection
├── data
│   └── malicious_phish.csv
├── notebooks
│   └── file1.ipynb
├── src
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

3. Start JupyterLab:

   ```bash
   jupyter lab
   ```

4. Open `notebooks/file1.ipynb` and run the cells in order.

## Dependencies

The project dependencies are listed in [`requirements.txt`](requirements.txt) and currently include:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `ipython`
- `ipykernel`
- `jupyterlab`
- `nbconvert`
- `nbformat`

## Current Scope

This repository currently focuses on analytical reporting: acquisition, understanding, cleaning, EDA, reproducibility, and statistical interpretation of malicious URL patterns. It does not yet include model training, validation, or deployment.
