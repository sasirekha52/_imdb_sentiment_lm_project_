from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Config:
    data_path: Path = PROJECT_ROOT / "data" / "raw" / "IMDB Dataset.csv"
    model_name: str = "distilbert/distilbert-base-uncased"
    model_output_dir: Path = PROJECT_ROOT / "models" / "distilbert-imdb-sentiment"
    report_dir: Path = PROJECT_ROOT / "reports"
    seed: int = 42
    max_length: int = 256
    test_size: float = 0.10
    validation_size: float = 0.10
    num_labels: int = 2

CONFIG = Config()
