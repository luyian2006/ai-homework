"""Text cleaning and TF-IDF feature construction.

TF-IDF must be fit on training data only. This project does that by putting the
vectorizer inside an sklearn Pipeline in models.py.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

def strip_email_headers(text: str) -> str:
    """Remove common newsgroup metadata lines from one document."""
    skipped_prefixes = (
        "from:",
        "subject:",
        "organization:",
        "lines:",
        "nntp-posting-host:",
        "reply-to:",
        "article-i.d.:",
        "keywords:",
        "distribution:",
    )

    kept_lines = []
    for line in text.splitlines():
        if not line.lower().startswith(skipped_prefixes):
            kept_lines.append(line)
    return "\n".join(kept_lines).lower()


def build_vectorizer(tfidf_params: dict) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer from the experiment configuration."""
    return TfidfVectorizer(
        preprocessor=strip_email_headers,
        **tfidf_params,
    )


def build_feature_transformer(feature_mode: str = "word"):
    """Build the requested sparse text representation."""
    if feature_mode == "word":
        from config import BASELINE_TFIDF_PARAMS

        return build_vectorizer(BASELINE_TFIDF_PARAMS)

    if feature_mode == "word_tuned":
        from config import TUNED_TFIDF_PARAMS

        return build_vectorizer(TUNED_TFIDF_PARAMS)

    raise ValueError(f"Unknown feature mode: {feature_mode}")
