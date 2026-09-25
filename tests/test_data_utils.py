import pandas as pd
from src.data_utils import load_and_validate, stratified_split

def test_validation_and_split(tmp_path):
    path = tmp_path / "mini.csv"
    pd.DataFrame({
        "review": [f"review {i}" for i in range(20)],
        "sentiment": ["positive", "negative"] * 10,
    }).to_csv(path, index=False)

    df = load_and_validate(path)
    train, val, test = stratified_split(df, seed=42, test_size=0.2, validation_size=0.2)

    assert set(df["label"]) == {0, 1}
    assert len(train) + len(val) + len(test) == len(df)
    assert train["label"].nunique() == 2
