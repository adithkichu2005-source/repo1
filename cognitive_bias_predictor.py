"""
Cognitive Bias Predictor
Trains a machine learning model to classify cognitive biases from text input.
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

MODEL_PATH = "cognitive_bias_model.pkl"
DATASET_PATH = "cognitive_bias_dataset.csv"


def load_data(dataset_path=DATASET_PATH):
    """Load and return the dataset."""
    df = pd.read_csv(dataset_path)
    return df["Sentence"].tolist(), df["CognitiveBias"].tolist()


def build_pipeline():
    """Build and return the sklearn Pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def train(dataset_path=DATASET_PATH, model_path=MODEL_PATH):
    """Train the model and save it to disk."""
    texts, labels = load_data(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    return pipeline


def load_model(model_path=MODEL_PATH):
    """Load and return the saved model, training it first if not found."""
    if not os.path.exists(model_path):
        print("Model not found. Training now...")
        return train(model_path=model_path)
    return joblib.load(model_path)


def predict(text, model=None, model_path=MODEL_PATH):
    """
    Predict the cognitive bias for a single text string.

    Returns a dict with keys 'bias' and 'confidence'.
    """
    if model is None:
        model = load_model(model_path)

    bias = str(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    confidence = float(max(probabilities))
    return {"bias": bias, "confidence": round(confidence, 4)}


def predict_batch(texts, model=None, model_path=MODEL_PATH):
    """
    Predict cognitive biases for a list of texts.

    Returns a list of dicts with keys 'text', 'bias', and 'confidence'.
    """
    if model is None:
        model = load_model(model_path)

    biases = model.predict(texts)
    probabilities = model.predict_proba(texts)

    results = []
    for text, bias, probs in zip(texts, biases, probabilities):
        results.append({
            "text": text,
            "bias": str(bias),
            "confidence": round(float(max(probs)), 4),
        })
    return results


if __name__ == "__main__":
    train()
