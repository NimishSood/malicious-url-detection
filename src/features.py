from __future__ import annotations

from pathlib import Path
from urllib.parse import SplitResult, urlsplit
import hashlib
import re

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

RANDOM_STATE = 42
EXPECTED_LABELS = ("benign", "phishing", "malware", "defacement")
SUSPICIOUS_TOKENS = ("login", "verify", "account", "secure", "update", "bank")
MODELING_SAMPLE_SIZE = 80_000
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
IP_HOST_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")

CORE_NUMERIC_FEATURES = [
    "url_length",
    "digit_count",
    "dot_count",
    "hyphen_count",
    "slash_count",
    "host_length",
    "path_length",
    "query_length",
    "path_depth",
    "subdomain_count",
]
CORE_BINARY_FEATURES = [
    "has_query",
    "has_scheme",
    "has_ip_host",
    "has_at_symbol",
    "has_percent_encoding",
]
TOKEN_FEATURES = [f"token_{token}" for token in SUSPICIOUS_TOKENS]
CORE_FEATURES = CORE_NUMERIC_FEATURES + CORE_BINARY_FEATURES
EXTENDED_FEATURES = CORE_FEATURES + TOKEN_FEATURES

LABEL_NORMALIZATION = {
    "benign": "benign",
    "phishing": "phishing",
    "malware": "malware",
    "defacement": "defacement",
    "0": "class_0",
    "1": "class_1",
    "2": "class_2",
    "3": "class_3",
}


def resolve_repo_root(start: Path | None = None) -> Path:
    base = (start or Path.cwd()).resolve()
    for candidate in (base, *base.parents):
        if (candidate / "data" / "malicious_phish.csv").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root containing data/malicious_phish.csv.")


def resolve_data_path(start: Path | None = None) -> Path:
    return resolve_repo_root(start) / "data" / "malicious_phish.csv"


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    data_path = (path or resolve_data_path()).resolve()
    df = pd.read_csv(data_path, dtype="string", engine="python", on_bad_lines="skip")

    normalized_cols = {str(col).strip().lower(): col for col in df.columns}
    if not {"url", "type"}.issubset(normalized_cols):
        raise ValueError(f"Expected url/type columns, found {list(df.columns)}")

    out = df[[normalized_cols["url"], normalized_cols["type"]]].copy()
    out.columns = ["url", "type"]
    out["url"] = out["url"].astype("string")
    out["type"] = out["type"].astype("string")

    provenance = {
        "data_path": str(data_path),
        "file_size_mb": round(data_path.stat().st_size / (1024 * 1024), 3),
        "sha256": file_sha256(data_path),
        "rows_loaded": int(len(out)),
        "columns_loaded": list(out.columns),
    }
    return out, provenance


def normalize_label(value) -> str:
    if pd.isna(value):
        return ""
    label = str(value).strip().lower()
    return LABEL_NORMALIZATION.get(label, label)


def run_cleaning_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    audit: list[dict[str, object]] = []

    def record(step: str, before: int, after: int) -> None:
        removed = before - after
        audit.append(
            {
                "step": step,
                "rows_before": before,
                "rows_after": after,
                "rows_removed": removed,
                "pct_removed": round((removed / before) * 100, 4) if before else 0.0,
            }
        )

    before = len(work)
    work["url"] = work["url"].astype("string").str.strip()
    work["type"] = work["type"].map(normalize_label).astype("string")
    record("normalize formatting and labels", before, len(work))

    before = len(work)
    work = work[work["url"].notna() & work["type"].notna()]
    record("drop missing url/type", before, len(work))

    before = len(work)
    work = work[(work["url"].str.len() > 0) & (work["type"].str.len() > 0)]
    record("drop empty url/type after strip", before, len(work))

    before = len(work)
    work = work[work["type"].isin(EXPECTED_LABELS)]
    record("keep expected labels only", before, len(work))

    before = len(work)
    work = work[~work["url"].str.contains(r"\s", regex=True, na=False)]
    record("remove URLs containing whitespace", before, len(work))

    before = len(work)
    work = work[~work["url"].str.contains(r"[\x00-\x1f\x7f]", regex=True, na=False)]
    record("remove URLs containing control chars", before, len(work))

    before = len(work)
    work = work.drop_duplicates()
    record("remove exact duplicate rows", before, len(work))

    before = len(work)
    conflicting_urls = work.groupby("url")["type"].nunique().loc[lambda s: s > 1].index
    work = work[~work["url"].isin(conflicting_urls)]
    record("remove duplicate URLs with conflicting labels", before, len(work))

    return work.reset_index(drop=True), pd.DataFrame(audit)


def summarize_label_distribution(df: pd.DataFrame, label_col: str = "type") -> pd.DataFrame:
    summary = (
        df[label_col]
        .value_counts(dropna=False)
        .rename_axis(label_col)
        .reset_index(name="count")
    )
    summary["percentage"] = (summary["count"] / summary["count"].sum() * 100).round(2)
    return summary


def sample_for_modeling(
    df: pd.DataFrame,
    sample_size: int = MODELING_SAMPLE_SIZE,
    stratify_col: str = "type",
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    if sample_size >= len(df):
        return df.copy().reset_index(drop=True)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=sample_size,
        random_state=random_state,
    )
    sample_idx, _ = next(splitter.split(df, df[stratify_col]))
    return df.iloc[sample_idx].reset_index(drop=True)


def split_url(url: str) -> SplitResult:
    candidate = url if SCHEME_RE.match(url) else f"http://{url}"
    return urlsplit(candidate)


def extract_host(url: str) -> str:
    try:
        return split_url(url).hostname or ""
    except ValueError:
        return ""


def extract_path(url: str) -> str:
    try:
        return split_url(url).path or ""
    except ValueError:
        return ""


def extract_query(url: str) -> str:
    try:
        return split_url(url).query or ""
    except ValueError:
        return ""


def extract_tld(host: str) -> str:
    if "." not in host:
        return "(none)"
    return host.rsplit(".", 1)[-1].lower()


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["host"] = features["url"].map(extract_host)
    features["path"] = features["url"].map(extract_path)
    features["query"] = features["url"].map(extract_query)
    features["tld"] = features["host"].map(extract_tld)

    features["url_length"] = features["url"].str.len()
    features["digit_count"] = features["url"].str.count(r"\d")
    features["dot_count"] = features["url"].str.count(r"\.")
    features["hyphen_count"] = features["url"].str.count(r"-")
    features["slash_count"] = features["url"].str.count(r"/")
    features["host_length"] = features["host"].str.len()
    features["path_length"] = features["path"].str.len()
    features["query_length"] = features["query"].str.len()
    features["path_depth"] = features["path"].str.count(r"/")
    features["subdomain_count"] = features["host"].str.count(r"\.")

    features["has_query"] = features["url"].str.contains(r"\?", regex=True, na=False).astype(int)
    features["has_scheme"] = features["url"].str.contains(r"^https?://", regex=True, case=False, na=False).astype(int)
    features["has_ip_host"] = features["host"].str.match(IP_HOST_RE, na=False).astype(int)
    features["has_at_symbol"] = features["url"].str.contains("@", regex=False, na=False).astype(int)
    features["has_percent_encoding"] = features["url"].str.contains("%", regex=False, na=False).astype(int)

    for token in SUSPICIOUS_TOKENS:
        features[f"token_{token}"] = features["url"].str.contains(
            fr"(?i){re.escape(token)}",
            regex=True,
            na=False,
        ).astype(int)

    return features


def get_feature_sets() -> dict[str, list[str]]:
    return {
        "core": CORE_FEATURES.copy(),
        "extended": EXTENDED_FEATURES.copy(),
    }


def make_binary_target(df: pd.DataFrame, source_col: str = "type") -> pd.Series:
    return df[source_col].where(df[source_col].eq("benign"), other="malicious")

