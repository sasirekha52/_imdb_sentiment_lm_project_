from pathlib import Path
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def build_baseline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=150_000,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(max_iter=1000, C=4.0)),
    ])

def train_baseline(train_df, test_df, output_path=None):
    model = build_baseline()
    model.fit(train_df["review"], train_df["label"])
    pred = model.predict(test_df["review"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_df["label"], pred, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(test_df["label"], pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
    return model, metrics, pred
