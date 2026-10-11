"""Metrics, experiment execution, and validation-result selection.

Use validation data for model selection. Save the final test prediction only
after the best validation configuration is chosen.
"""

import time

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


def evaluate_predictions(y_true, y_pred) -> dict:
    """Return the main metrics used by the report."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def run_one_experiment(
    config: dict,
    pipeline: Pipeline,
    x_train: list[str],
    y_train,
    x_valid: list[str],
    y_valid,
) -> tuple[dict, Pipeline]:
    """Fit one candidate pipeline and return its metrics and fitted model."""
    start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_valid)
    elapsed = time.perf_counter() - start

    metrics = evaluate_predictions(y_valid, predictions)
    result = {
        "config": config["name"],
        "model": config["model"],
            "feature_mode": config.get("feature_mode", "word"),
        "params": config["params"],
        "accuracy": round(metrics["accuracy"], 6),
        "macro_f1": round(metrics["macro_f1"], 6),
        "seconds": round(elapsed, 2),
    }
    return result, pipeline


def choose_best_result(results: pd.DataFrame, metric: str = "macro_f1") -> str:
    """Return the config name with the highest selected validation metric."""
    if metric not in results.columns:
        raise ValueError(f"Metric {metric!r} is not in the result table.")
    return results.sort_values(metric, ascending=False).iloc[0]["config"]


def print_report(y_true, y_pred) -> None:
    """Print a per-class report for the selected validation model."""
    print(classification_report(y_true, y_pred, digits=4))
