
"""Load APACHE IV and IVa ICU mortality predictions."""

from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parent
APACHE_PATH = ROOT / "data" / "apachePatientResult.csv"

def load_apache(patient_id):
    columns = [
        "patientunitstayid",
        "apacheversion",
        "apachescore",
        "predictedicumortality"
    ]
    df = pd.read_csv(APACHE_PATH, usecols=columns)
    rows = df[df["patientunitstayid"] == int(patient_id)]

    results = {}
    for _, row in rows.iterrows():
        version = str(row["apacheversion"])
        probability = pd.to_numeric(
            row["predictedicumortality"], errors="coerce"
        )
        if version not in ("IV", "IVa") or not math.isfinite(probability):
            continue
        if not 0 <= probability <= 1:
            continue
        if version in results:
            raise ValueError(f"Duplicate APACHE version: {version}")

        results[version] = {
            "version": version,
            "score": int(row["apachescore"]) if pd.notna(row["apachescore"]) else None,
            "mortality_probability": float(probability)
        }

    return results
