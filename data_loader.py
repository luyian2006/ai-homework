"""Functions for reading the CSV files and creating a validation split.

Experiment 1 requires students to create their own validation set from the
provided labeled training data. Do not use the unlabeled test set for tuning.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(
    train_file: Path,
    test_file: Path,
) -> tuple[list[str], pd.Series, list[str]]:
    """Return labeled texts, labels, and unlabeled test texts."""
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    required_train_columns = {"text", "target"}
    required_test_columns = {"text"}
    if not required_train_columns.issubset(train_df.columns):
        raise ValueError(f"Training CSV must contain columns: {required_train_columns}")
    if not required_test_columns.issubset(test_df.columns):
        raise ValueError(f"Test CSV must contain columns: {required_test_columns}")

    # fillna avoids turning missing values into the literal string "nan".
    texts = train_df["text"].fillna("").astype(str).tolist()
    labels = train_df["target"].astype(int)
    test_texts = test_df["text"].fillna("").astype(str).tolist()

    return texts, labels, test_texts


def split_data(
    texts: list[str],
    labels: pd.Series,
    validation_size: float,
    random_state: int,
) -> tuple[list[str], list[str], pd.Series, pd.Series]:
    """Create a stratified train/validation split."""
    return train_test_split(
        texts,
        labels,
        test_size=validation_size,
        random_state=random_state,
        stratify=labels,
    )
