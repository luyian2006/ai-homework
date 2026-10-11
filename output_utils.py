"""Functions for writing predictions and experiment tables.

The submission file for Experiment 1 should be one column, no header, no index,
and should preserve the original order of test_data_unlabeled.csv.
"""

from pathlib import Path

import pandas as pd


def save_predictions(predictions, output_file: Path) -> None:
    """Write the required one-column, no-header submission file."""
    pd.DataFrame(predictions).to_csv(
        output_file,
        index=False,
        header=False,
    )


def save_validation_results(results: pd.DataFrame, output_file: Path) -> None:
    """Write model-comparison results for the report."""
    results.sort_values(
        by=["macro_f1", "accuracy"],
        ascending=False,
    ).to_csv(output_file, index=False)
