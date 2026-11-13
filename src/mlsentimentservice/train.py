import logging

import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mlsentimentservice.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_data() -> tuple[list[str], list[int]]:
    """Download (if needed) and load the NLTK movie_reviews dataset."""
    nltk.download("movie_reviews", quiet=True)
    from nltk.corpus import movie_reviews

    texts: list[str] = []
    labels: list[int] = []

    for fileid in movie_reviews.fileids():
        label = 1 if fileid.startswith("pos") else 0
        texts.append(movie_reviews.raw(fileid))
        labels.append(label)

    logger.info(
        "Loaded %d reviews (pos=%d, neg=%d)",
        len(texts),
        sum(labels),
        len(labels) - sum(labels),
    )
    return texts, labels


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10_000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1_000, C=1.0, random_state=42),
            ),
        ]
    )


def train() -> Pipeline:
    texts, labels = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    pipeline = build_pipeline()
    logger.info("Training pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Test accuracy: %.4f", acc)
    logger.info(
        "\n%s",
        classification_report(y_test, y_pred, target_names=["negative", "positive"]),
    )
    return pipeline


def main() -> None:
    pipeline = train()

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "version": settings.model_version},
        settings.model_path,
    )
    logger.info("Model artifact saved to %s", settings.model_path)


if __name__ == "__main__":
    main()