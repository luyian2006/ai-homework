"""Model factories and leakage-safe sklearn pipelines.

The assignment asks for classical text-classification models and validation-set
comparison. This file creates those models without fitting them.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from config import RANDOM_STATE
from preprocessing import build_feature_transformer


def build_classifier(model_name: str, model_params: dict):
    """Build one classifier from a short model name."""
    if model_name == "nb":
        return MultinomialNB(**model_params)

    if model_name == "linear_svm":
        return SVC(kernel="linear", random_state=RANDOM_STATE, **model_params)

    if model_name == "logistic_regression":
        return LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=2000,
            solver="liblinear",
            **model_params,
        )

    if model_name == "mlp":
        return MLPClassifier(
            random_state=RANDOM_STATE,
            early_stopping=True,
            **model_params,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def build_pipeline(
    model_name: str,
    model_params: dict,
    feature_mode: str = "word",
) -> Pipeline:
    """
    Combine TF-IDF and the classifier.

    Because the vectorizer is inside Pipeline, fit() learns vocabulary only
    from the data passed to that fit call.
    """
    return Pipeline(
        [
            ("features", build_feature_transformer(feature_mode)),
            ("classifier", build_classifier(model_name, model_params)),
        ]
    )
