from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def overall_metrics_frame(
    y_true,
    y_pred,
    labels: list[str],
    model_name: str,
    task_name: str,
    positive_label: str | None = None,
) -> pd.DataFrame:
    metrics = {
        "task": task_name,
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }

    if positive_label is not None:
        metrics["precision"] = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    else:
        metrics["precision"] = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    return pd.DataFrame([metrics]).round(4)


def per_class_metrics_frame(
    y_true,
    y_pred,
    labels: list[str],
    model_name: str,
    task_name: str,
) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    frame = pd.DataFrame(
        {
            "task": task_name,
            "model": model_name,
            "class": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    return frame.round(4)


def confusion_matrix_frame(
    y_true,
    y_pred,
    labels: list[str],
) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.index.name = "actual"
    frame.columns.name = "predicted"
    return frame


def grid_search_results_frame(grid_search, model_name: str, task_name: str) -> pd.DataFrame:
    results = pd.DataFrame(grid_search.cv_results_)
    keep_cols = [
        col
        for col in results.columns
        if col.startswith("param_")
        or col.startswith("mean_test_")
        or col.startswith("std_test_")
        or col.startswith("rank_test_")
        or col == "mean_fit_time"
    ]
    frame = results[keep_cols].copy()
    frame.insert(0, "task", task_name)
    frame.insert(1, "model", model_name)
    refit_metric = grid_search.refit
    frame = frame.sort_values(f"rank_test_{refit_metric}").reset_index(drop=True)
    return frame


def logistic_coefficients_frame(
    fitted_pipeline,
    class_labels: list[str],
    numeric_features: list[str],
    top_n_per_class: int | None = None,
) -> pd.DataFrame:
    prep = fitted_pipeline.named_steps["prep"]
    model = fitted_pipeline.named_steps["model"]
    feature_names = prep.get_feature_names_out().tolist()
    coef = np.atleast_2d(model.coef_)
    classes = class_labels if len(class_labels) > 2 else [class_labels[-1]]

    rows: list[dict[str, object]] = []
    for class_name, weights in zip(classes, coef):
        class_frame = pd.DataFrame(
            {
                "class": class_name,
                "feature": feature_names,
                "coefficient": weights,
            }
        )
        class_frame["abs_coefficient"] = class_frame["coefficient"].abs()
        class_frame["odds_ratio"] = np.exp(class_frame["coefficient"])
        class_frame["direction"] = np.where(class_frame["coefficient"] >= 0, "positive", "negative")
        class_frame["unit"] = np.where(
            class_frame["feature"].isin(numeric_features),
            "per 1 SD increase",
            "feature present vs absent",
        )
        class_frame = class_frame.sort_values("abs_coefficient", ascending=False)
        if top_n_per_class is not None:
            class_frame = class_frame.head(top_n_per_class)
        rows.extend(class_frame.to_dict(orient="records"))

    return pd.DataFrame(rows).round(4)


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_confusion_matrices(
    confusion_matrices: dict[str, pd.DataFrame],
    output_dir: Path,
    prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, matrix in confusion_matrices.items():
        filename = f"{prefix}_confusion_matrix_{slugify(model_name)}.csv"
        matrix.to_csv(output_dir / filename)


def save_metadata(metadata: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

