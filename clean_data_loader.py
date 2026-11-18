
"""Load the 53 structured 24h features for an ICU episode."""

from pathlib import Path
from functools import lru_cache
import pandas as pd
from ml_model.predict import load_model

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "ml_model" / "ml_features_24h.csv"


@lru_cache(maxsize=1)
def load_dataset():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        dtype={"patientunitstayid": "string"},
        low_memory=False
    )

    if df["patientunitstayid"].duplicated().any():
        raise ValueError("Duplicate ICU episode IDs.")

    return df.set_index("patientunitstayid")


def clean_data_loader(patient_id):
    """Return the ML feature vector for one ICU episode."""
    df = load_dataset()
    episode_id = str(patient_id)

    if episode_id not in df.index:
        raise ValueError(
            f"Episode {episode_id} not found in the 24h dataset."
        )

    row = df.loc[episode_id]
    _, feature_names = load_model()

    features = {
        name: None if pd.isna(row[name]) else row[name]
        for name in feature_names
    }

    return features
