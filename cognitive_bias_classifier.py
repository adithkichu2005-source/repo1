"""
Cognitive Bias Classifier

This script loads cognitive_bias_dataset.csv, preprocesses the text using NLTK,
extracts TF-IDF features, trains a Logistic Regression classifier to predict
cognitive bias types, evaluates its performance, and saves the trained model.

Supported bias categories:
    overgeneralization, catastrophizing, black_and_white_thinking,
    mind_reading, emotional_reasoning, no_bias
"""

import os
import logging

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK resource bootstrap
# ---------------------------------------------------------------------------
NLTK_RESOURCES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]


def _download_nltk_resources() -> None:
    """Download required NLTK resources if they are not already present."""
    for resource in NLTK_RESOURCES:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                logger.info("Downloading NLTK resource: %s", resource)
                nltk.download(resource, quiet=True)
    # Populate the module-level stop words set after resources are available
    global _STOP_WORDS
    _STOP_WORDS = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
_lemmatizer = WordNetLemmatizer()
_STOP_WORDS: set = set()  # populated after NLTK resources are available


def preprocess_text(text: str) -> str:
    """Tokenize, lowercase, remove stopwords, and lemmatize *text*.

    Parameters
    ----------
    text:
        Raw sentence to preprocess.

    Returns
    -------
    str
        Space-joined string of cleaned tokens.
    """
    tokens = word_tokenize(text.lower())
    tokens = [
        _lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in _STOP_WORDS
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

DATA_FILE = os.path.join(os.path.dirname(__file__), "cognitive_bias_dataset.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "cognitive_bias_model.pkl")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from *path* and validate required columns.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing at least ``Sentence`` and ``CognitiveBias`` columns.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the required columns are missing from the file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    required_columns = {"Sentence", "CognitiveBias"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    logger.info("Loaded %d samples from '%s'.", len(df), path)
    return df


def train_and_evaluate(data_path: str = DATA_FILE, model_path: str = MODEL_FILE) -> dict:
    """End-to-end training, evaluation, and model persistence.

    Parameters
    ----------
    data_path:
        Path to the CSV dataset.
    model_path:
        Destination path for the saved model artifact.

    Returns
    -------
    dict
        Dictionary with keys ``accuracy``, ``report``, and ``model``.
    """
    # Ensure NLTK resources (and _STOP_WORDS) are available
    _download_nltk_resources()

    # 1. Load data
    df = load_data(data_path)

    # 2. Preprocess text
    logger.info("Preprocessing text…")
    df["processed_text"] = df["Sentence"].apply(preprocess_text)

    # 3. Prepare features and labels
    X = df["processed_text"]
    y = df["CognitiveBias"]

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        "Split: %d training samples, %d test samples.", len(X_train), len(X_test)
    )

    # 5. Build pipeline: TF-IDF → Logistic Regression
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
                ),
            ),
        ]
    )

    logger.info("Training Logistic Regression classifier…")
    pipeline.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logger.info("Accuracy: %.4f", accuracy)
    logger.info("Classification Report:\n%s", report)

    # 7. Save model
    joblib.dump(pipeline, model_path)
    logger.info("Model saved to '%s'.", model_path)

    return {"accuracy": accuracy, "report": report, "model": pipeline}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _download_nltk_resources()
    results = train_and_evaluate()
    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results["report"])
