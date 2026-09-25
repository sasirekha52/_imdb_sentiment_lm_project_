from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {"review", "sentiment"}
LABEL_MAP = {"negative": 0, "positive": 1}
ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
LABEL2ID = {"NEGATIVE": 0, "POSITIVE": 1}

def load_and_validate(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[["review", "sentiment"]].copy()
    df["review"] = df["review"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df = df[df["review"].str.len() > 0]
    df = df[df["sentiment"].isin(LABEL_MAP)]
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)
    df["label"] = df["sentiment"].map(LABEL_MAP).astype(int)

    if df.empty:
        raise ValueError("No usable rows remain after validation.")

    return df

def stratified_split(df, seed=42, test_size=0.10, validation_size=0.10):
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    relative_val = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=train_val["label"],
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)
