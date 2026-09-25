from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_predictions(y_true, y_pred, labels=(0, 1), target_names=("NEGATIVE", "POSITIVE")):
    report = classification_report(
        y_true, y_pred, labels=list(labels), target_names=list(target_names),
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return report, cm

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")

def measure_latency(predict_fn, texts, repeats=1):
    start = time.perf_counter()
    output = predict_fn(texts)
    elapsed = time.perf_counter() - start
    return {
        "samples": len(texts),
        "seconds": elapsed,
        "milliseconds_per_sample": (elapsed / max(len(texts), 1)) * 1000,
        "predictions": output,
    }
