from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "sentiment.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "sentiment_model.joblib"

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    required_columns = {"text", "sentiment"}
    if not required_columns.issubset(df.columns):
        raise ValueError("CSV must contain text and sentiment columns.")

    df = df.dropna(subset=["text", "sentiment"])
    df["text"] = df["text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["sentiment"] != "")]

    print("Dataset size:", len(df))
    print("\nClass distribution:")
    print(df["sentiment"].value_counts())

    X = df["text"]
    y = df["sentiment"]

    if y.nunique() < 2:
        raise ValueError("At least two sentiment classes are required.")
    if y.value_counts().min() < 2:
        raise ValueError("Each class needs at least two examples for stratified splitting.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = sorted(y.unique())
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_precision": precision_score(y_test, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "labels": labels
    }

    print("\nEvaluation results")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nClassification report:")
    report = classification_report(y_test, predictions, zero_division=0)
    print(report)

    cm = confusion_matrix(y_test, predictions, labels=labels)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted sentiment")
    plt.ylabel("Actual sentiment")
    plt.title("Sentiment Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    joblib.dump(model, MODEL_PATH)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=4), encoding="utf-8"
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    print("\nSaved model:", MODEL_PATH)
    print("Saved evaluation files in outputs/")

if __name__ == "__main__":
    main()
