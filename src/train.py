from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import (
    confusion_matrix_frame,
    grid_search_results_frame,
    logistic_coefficients_frame,
    overall_metrics_frame,
    per_class_metrics_frame,
    save_confusion_matrices,
    save_frame,
    save_metadata,
)
from src.features import (
    CORE_BINARY_FEATURES,
    CORE_NUMERIC_FEATURES,
    EXTENDED_FEATURES,
    MODELING_SAMPLE_SIZE,
    RANDOM_STATE,
    build_feature_frame,
    get_feature_sets,
    load_dataset,
    make_binary_target,
    run_cleaning_pipeline,
    sample_for_modeling,
    summarize_label_distribution,
)

BINARY_LABELS = ["benign", "malicious"]
MULTICLASS_LABELS = ["benign", "defacement", "malware", "phishing"]
BINARY_F1_SCORER = make_scorer(f1_score, pos_label="malicious")


def build_preprocessor(numeric_features: list[str], binary_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            ("bin", "passthrough", binary_features),
        ],
        verbose_feature_names_out=False,
    )


def logistic_search(
    numeric_features: list[str],
    binary_features: list[str],
    scoring: dict[str, str],
    refit_metric: str,
    cv_folds: int,
) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("prep", build_preprocessor(numeric_features, binary_features)),
            ("model", LogisticRegression(max_iter=1500, solver="lbfgs")),
        ]
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid={
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
        scoring=scoring,
        refit=refit_metric,
        cv=cv_folds,
        n_jobs=-1,
    )


def knn_search(
    numeric_features: list[str],
    binary_features: list[str],
    scoring: dict[str, str],
    refit_metric: str,
    cv_folds: int,
) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("prep", build_preprocessor(numeric_features, binary_features)),
            ("model", KNeighborsClassifier()),
        ]
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid={
            "model__n_neighbors": [5, 11, 21, 31],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        scoring=scoring,
        refit=refit_metric,
        cv=cv_folds,
        n_jobs=-1,
    )


def feature_set_comparison_frame(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
) -> tuple[pd.DataFrame, str]:
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": BINARY_F1_SCORER,
    }
    rows: list[dict[str, object]] = []
    feature_sets = get_feature_sets()
    best_feature_set = "core"
    best_score = float("-inf")

    for feature_set_name, feature_list in feature_sets.items():
        numeric_features = [feature for feature in feature_list if feature in CORE_NUMERIC_FEATURES]
        binary_features = [feature for feature in feature_list if feature not in numeric_features]
        search = logistic_search(
            numeric_features=numeric_features,
            binary_features=binary_features,
            scoring=scoring,
            refit_metric="balanced_accuracy",
            cv_folds=cv_folds,
        )
        search.fit(X_train[feature_list], y_train)
        row = {
            "feature_set": feature_set_name,
            "cv_accuracy": search.cv_results_["mean_test_accuracy"][search.best_index_],
            "cv_balanced_accuracy": search.cv_results_["mean_test_balanced_accuracy"][search.best_index_],
            "cv_f1": search.cv_results_["mean_test_f1"][search.best_index_],
            "best_C": search.best_params_["model__C"],
            "best_class_weight": search.best_params_["model__class_weight"],
        }
        rows.append(row)

        if row["cv_balanced_accuracy"] > best_score:
            best_feature_set = feature_set_name
            best_score = row["cv_balanced_accuracy"]

    comparison = pd.DataFrame(rows).sort_values("cv_balanced_accuracy", ascending=False).reset_index(drop=True)
    comparison["selected"] = comparison["feature_set"].eq(best_feature_set)
    return comparison.round(4), best_feature_set


def evaluate_model_bundle(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    labels: list[str],
    task_name: str,
    positive_label: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    overall_frames = []
    class_frames = []
    confusion_frames: dict[str, pd.DataFrame] = {}

    for model_name, fitted_model in models.items():
        predictions = fitted_model.predict(X_test)
        overall_frames.append(
            overall_metrics_frame(
                y_true=y_test,
                y_pred=predictions,
                labels=labels,
                model_name=model_name,
                task_name=task_name,
                positive_label=positive_label,
            )
        )
        class_frames.append(
            per_class_metrics_frame(
                y_true=y_test,
                y_pred=predictions,
                labels=labels,
                model_name=model_name,
                task_name=task_name,
            )
        )
        confusion_frames[model_name] = confusion_matrix_frame(
            y_true=y_test,
            y_pred=predictions,
            labels=labels,
        )

    overall = pd.concat(overall_frames, ignore_index=True)
    per_class = pd.concat(class_frames, ignore_index=True)
    return overall, per_class, confusion_frames


def run_binary_workflow(
    feature_frame: pd.DataFrame,
    cv_folds: int,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    task_name = "binary_benign_vs_malicious"
    feature_sets = get_feature_sets()
    working = feature_frame.copy()
    working["binary_target"] = make_binary_target(working)

    X_train, X_test, y_train, y_test = train_test_split(
        working,
        working["binary_target"],
        test_size=test_size,
        stratify=working["binary_target"],
        random_state=random_state,
    )

    feature_comparison, selected_feature_set = feature_set_comparison_frame(
        X_train=X_train,
        y_train=y_train,
        cv_folds=cv_folds,
    )

    selected_features = feature_sets[selected_feature_set]
    numeric_features = [feature for feature in selected_features if feature in CORE_NUMERIC_FEATURES]
    binary_features = [feature for feature in selected_features if feature not in numeric_features]

    logistic = logistic_search(
        numeric_features=numeric_features,
        binary_features=binary_features,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1": BINARY_F1_SCORER,
        },
        refit_metric="balanced_accuracy",
        cv_folds=cv_folds,
    )
    logistic.fit(X_train[selected_features], y_train)

    knn = knn_search(
        numeric_features=numeric_features,
        binary_features=binary_features,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1": BINARY_F1_SCORER,
        },
        refit_metric="balanced_accuracy",
        cv_folds=cv_folds,
    )
    knn.fit(X_train[selected_features], y_train)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train[selected_features], y_train)

    models = {
        "Majority baseline": baseline,
        "Logistic Regression": logistic.best_estimator_,
        "k-NN": knn.best_estimator_,
    }
    overall, per_class, confusion = evaluate_model_bundle(
        models=models,
        X_test=X_test[selected_features],
        y_test=y_test,
        labels=BINARY_LABELS,
        task_name=task_name,
        positive_label="malicious",
    )

    coefficients = logistic_coefficients_frame(
        fitted_pipeline=logistic.best_estimator_,
        class_labels=BINARY_LABELS,
        numeric_features=numeric_features,
        top_n_per_class=None,
    )

    return {
        "task_name": task_name,
        "selected_feature_set": selected_feature_set,
        "selected_features": selected_features,
        "feature_set_comparison": feature_comparison,
        "cv_results": pd.concat(
            [
                grid_search_results_frame(logistic, "Logistic Regression", task_name),
                grid_search_results_frame(knn, "k-NN", task_name),
            ],
            ignore_index=True,
        ),
        "test_metrics": overall.sort_values("balanced_accuracy", ascending=False).reset_index(drop=True),
        "per_class_metrics": per_class,
        "confusion_matrices": confusion,
        "logistic_coefficients": coefficients,
        "best_params": {
            "logistic": logistic.best_params_,
            "knn": knn.best_params_,
        },
    }


def run_multiclass_workflow(
    feature_frame: pd.DataFrame,
    selected_feature_set: str,
    cv_folds: int,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    task_name = "multiclass_url_type"
    feature_sets = get_feature_sets()
    selected_features = feature_sets[selected_feature_set]
    numeric_features = [feature for feature in selected_features if feature in CORE_NUMERIC_FEATURES]
    binary_features = [feature for feature in selected_features if feature not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame,
        feature_frame["type"],
        test_size=test_size,
        stratify=feature_frame["type"],
        random_state=random_state,
    )

    logistic = logistic_search(
        numeric_features=numeric_features,
        binary_features=binary_features,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
        },
        refit_metric="f1_macro",
        cv_folds=cv_folds,
    )
    logistic.fit(X_train[selected_features], y_train)

    knn = knn_search(
        numeric_features=numeric_features,
        binary_features=binary_features,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
        },
        refit_metric="f1_macro",
        cv_folds=cv_folds,
    )
    knn.fit(X_train[selected_features], y_train)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train[selected_features], y_train)

    models = {
        "Majority baseline": baseline,
        "Logistic Regression": logistic.best_estimator_,
        "k-NN": knn.best_estimator_,
    }
    overall, per_class, confusion = evaluate_model_bundle(
        models=models,
        X_test=X_test[selected_features],
        y_test=y_test,
        labels=MULTICLASS_LABELS,
        task_name=task_name,
    )

    coefficients = logistic_coefficients_frame(
        fitted_pipeline=logistic.best_estimator_,
        class_labels=MULTICLASS_LABELS,
        numeric_features=numeric_features,
        top_n_per_class=None,
    )

    return {
        "task_name": task_name,
        "selected_feature_set": selected_feature_set,
        "selected_features": selected_features,
        "cv_results": pd.concat(
            [
                grid_search_results_frame(logistic, "Logistic Regression", task_name),
                grid_search_results_frame(knn, "k-NN", task_name),
            ],
            ignore_index=True,
        ),
        "test_metrics": overall.sort_values("macro_f1", ascending=False).reset_index(drop=True),
        "per_class_metrics": per_class,
        "confusion_matrices": confusion,
        "logistic_coefficients": coefficients,
        "best_params": {
            "logistic": logistic.best_params_,
            "knn": knn.best_params_,
        },
    }


def save_results(
    output_dir: Path,
    metadata: dict[str, Any],
    cleaning_audit: pd.DataFrame,
    cleaned_summary: pd.DataFrame,
    modeling_summary: pd.DataFrame,
    binary_results: dict[str, Any],
    multiclass_results: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(metadata, output_dir / "run_metadata.json")
    save_frame(cleaning_audit, output_dir / "cleaning_audit.csv")
    save_frame(cleaned_summary, output_dir / "cleaned_class_distribution.csv")
    save_frame(modeling_summary, output_dir / "modeling_sample_class_distribution.csv")

    save_frame(binary_results["feature_set_comparison"], output_dir / "binary_feature_set_comparison.csv")
    save_frame(binary_results["cv_results"], output_dir / "binary_cv_results.csv")
    save_frame(binary_results["test_metrics"], output_dir / "binary_test_metrics.csv")
    save_frame(binary_results["per_class_metrics"], output_dir / "binary_per_class_metrics.csv")
    save_frame(binary_results["logistic_coefficients"], output_dir / "binary_logistic_coefficients.csv")
    save_confusion_matrices(binary_results["confusion_matrices"], output_dir, "binary")

    save_frame(multiclass_results["cv_results"], output_dir / "multiclass_cv_results.csv")
    save_frame(multiclass_results["test_metrics"], output_dir / "multiclass_test_metrics.csv")
    save_frame(multiclass_results["per_class_metrics"], output_dir / "multiclass_per_class_metrics.csv")
    save_frame(multiclass_results["logistic_coefficients"], output_dir / "multiclass_logistic_coefficients.csv")
    save_confusion_matrices(multiclass_results["confusion_matrices"], output_dir, "multiclass")


def run_full_workflow(
    output_dir: Path | None = None,
    sample_size: int = MODELING_SAMPLE_SIZE,
    cv_folds: int = 3,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    raw_df, provenance = load_dataset()
    cleaned_df, cleaning_audit = run_cleaning_pipeline(raw_df)
    cleaned_summary = summarize_label_distribution(cleaned_df, label_col="type")

    modeling_sample = sample_for_modeling(
        cleaned_df,
        sample_size=sample_size,
        stratify_col="type",
        random_state=random_state,
    )
    modeling_summary = summarize_label_distribution(modeling_sample, label_col="type")
    feature_frame = build_feature_frame(modeling_sample)

    binary_results = run_binary_workflow(
        feature_frame=feature_frame,
        cv_folds=cv_folds,
        test_size=test_size,
        random_state=random_state,
    )
    multiclass_results = run_multiclass_workflow(
        feature_frame=feature_frame,
        selected_feature_set=binary_results["selected_feature_set"],
        cv_folds=cv_folds,
        test_size=test_size,
        random_state=random_state,
    )

    metadata = {
        "random_state": random_state,
        "cv_folds": cv_folds,
        "test_size": test_size,
        "sample_size_requested": sample_size,
        "sample_size_used": int(len(modeling_sample)),
        "cleaned_rows": int(len(cleaned_df)),
        "binary_selected_feature_set": binary_results["selected_feature_set"],
        "binary_best_params": binary_results["best_params"],
        "multiclass_best_params": multiclass_results["best_params"],
        "provenance": provenance,
    }

    if output_dir is not None:
        save_results(
            output_dir=output_dir,
            metadata=metadata,
            cleaning_audit=cleaning_audit,
            cleaned_summary=cleaned_summary,
            modeling_summary=modeling_summary,
            binary_results=binary_results,
            multiclass_results=multiclass_results,
        )

    return {
        "metadata": metadata,
        "cleaning_audit": cleaning_audit,
        "cleaned_summary": cleaned_summary,
        "modeling_summary": modeling_summary,
        "feature_frame": feature_frame,
        "binary": binary_results,
        "multiclass": multiclass_results,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results = run_full_workflow(output_dir=repo_root / "results")
    print("Saved modelling outputs to", repo_root / "results")
    print("Binary selected feature set:", results["binary"]["selected_feature_set"])


if __name__ == "__main__":
    main()
