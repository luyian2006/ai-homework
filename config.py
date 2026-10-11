"""Central configuration for Experiment 1.

This file should contain the choices you will later explain in the report:
data paths, random seed, validation ratio, TF-IDF settings, models, and
hyperparameters.
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "train_data.csv"
TEST_FILE = BASE_DIR / "test_data_unlabeled.csv"
PREDICTION_FILE = BASE_DIR / "predictions.csv"
VALIDATION_RESULT_FILE = BASE_DIR / "validation_results.csv"
REPORT_FILE = BASE_DIR / "validation_report.md"
REPORT_TOP_ROWS = 3

# Charts for the experiment report.
MACRO_F1_CHART_FILE = BASE_DIR / "macro_f1_comparison.png"
CONFUSION_MATRIX_FILE = BASE_DIR / "confusion_matrix.png"
LOSS_CURVE_FILE = BASE_DIR / "mlp_loss_curve.png"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.2

BASELINE_TFIDF_PARAMS = {
    "lowercase": True,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "max_features": 30000,
    "sublinear_tf": True,
}

TUNED_TFIDF_PARAMS = {
    "lowercase": True,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 1,
    "max_df": 0.95,
    "max_features": 50000,
    "sublinear_tf": True,
}

# Kept as the default name for callers that do not need the baseline/tuned
# distinction.
TFIDF_PARAMS = BASELINE_TFIDF_PARAMS

def write_validation_report(
    results_file: Path = VALIDATION_RESULT_FILE,
    report_file: Path = REPORT_FILE,
    top_rows: int = REPORT_TOP_ROWS,
) -> None:
    """Write the best validation configurations and selection rationale."""
    import pandas as pd

    if top_rows < 1:
        raise ValueError("top_rows must be at least 1")

    results = pd.read_csv(results_file)
    required_columns = {"config", "model", "accuracy", "macro_f1"}
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        raise ValueError(f"Validation results missing columns: {sorted(missing_columns)}")
    if results.empty:
        raise ValueError("Validation results are empty")

    ranked = results.sort_values(
        by=["macro_f1", "accuracy"],
        ascending=False,
        kind="stable",
    ).head(top_rows)
    selected = ranked.iloc[0]

    columns = list(ranked.columns)

    def markdown_cell(value) -> str:
        return str(value).replace("|", r"\|").replace("\n", " ")

    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    table.extend(
        "| " + " | ".join(markdown_cell(row[column]) for column in columns) + " |"
        for _, row in ranked.iterrows()
    )

    report_file.write_text(
        "\n".join(
            [
                "# Validation Results",
                "",
                "## Best Configurations",
                "",
                *table,
                "",
                "## Model Selection",
                "",
                (
                    f"The selected configuration was `{selected['config']}` "
                    f"({selected['model']}) because it achieved the highest "
                    f"validation macro-F1 ({selected['macro_f1']}) among the "
                    "tested configurations. Accuracy was used as the tie-breaker."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )


EXPERIMENT_CONFIGS = [
    {"name": "nb_alpha_0.5", "model": "nb", "params": {"alpha": 0.5}},
    {"name": "nb_alpha_1.0", "model": "nb", "params": {"alpha": 1.0}},
    {"name": "svm_c_0.1", "model": "linear_svm", "params": {"C": 0.1}},
    {"name": "svm_c_1.0", "model": "linear_svm", "params": {"C": 1.0}},
    {"name": "svm_c_10.0", "model": "linear_svm", "params": {"C": 10.0}},
    {"name": "lr_c_0.1", "model": "logistic_regression", "params": {"C": 0.1}},
    {"name": "lr_c_1.0", "model": "logistic_regression", "params": {"C": 1.0}},
    {"name": "lr_c_10.0", "model": "logistic_regression", "params": {"C": 10.0}},
    {
        "name": "mlp_hidden_100",
        "model": "mlp",
        "params": {"hidden_layer_sizes": (100,), "alpha": 0.0001, "max_iter": 50},
    },
    {
        "name": "svm_tfidf_50k_min_df1_c_10.0",
        "model": "linear_svm",
        "params": {"C": 10.0},
        "feature_mode": "word_tuned",
    },
]
