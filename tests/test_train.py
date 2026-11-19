
import pandas as pd
from pathlib import Path


def test_dataset_exists():
    assert Path("data/dataset.csv").exists()


def test_dataset_has_target():
    df = pd.read_csv("data/dataset.csv")
    assert "target" in df.columns


def test_dataset_has_features():
    df = pd.read_csv("data/dataset.csv")
    assert len(df.columns) > 1


def test_dataset_is_not_empty():
    df = pd.read_csv("data/dataset.csv")
    assert len(df) > 0
