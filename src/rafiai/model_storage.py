
from pathlib import Path
import joblib


def get_model_path():
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"

    models_dir.mkdir(parents=True, exist_ok=True)

    return models_dir / "rafiai_model.joblib"


def model_exists():
    return get_model_path().exists()


def save_model(model, vectorizer):
    model_path = get_model_path()

    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer
        },
        model_path
    )


def load_model():
    model_path = get_model_path()

    if not model_path.exists():
        raise FileNotFoundError("Trained model not found.")

    return joblib.load(model_path)
