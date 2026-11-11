
from pathlib import Path

import pandas as pd


def load_dataset(file_name):
    """
    Loads a CSV dataset from the project's data directory.
    """

    project_root = Path(__file__).resolve().parents[2]
    file_path = project_root / "data" / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    dataset = pd.read_csv(file_path)

    required_columns = {"text", "label"}

    if not required_columns.issubset(dataset.columns):
        raise ValueError(
            f"The dataset must contain these columns: {required_columns}"
        )

    if dataset.empty:
        raise ValueError("The dataset is empty.")

    if dataset[["text", "label"]].isnull().any().any():
        raise ValueError("The dataset contains missing values.")

    return dataset
