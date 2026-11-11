
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "processed_data"
DATA_DIR = ROOT / "data"
OUTPUT_FILE = ROOT / "ml_model" / "ml_features_dynamic.csv"

PREDICTION_HOURS = [12, 16, 20, 24]

PERIODIC_SIGNALS = {
    "heartrate": "hr",
    "sao2": "sao2",
    "respiration": "rr"
}
APERIODIC_SIGNALS = {
    "noninvasivesystolic": "sbp",
    "noninvasivediastolic": "dbp",
    "noninvasivemean": "map"
}
LAB_NAMES = ["BUN", "creatinine", "glucose", "potassium", "sodium"]
CLINICAL_FEATURES = ["age", "gender", "unit_type", "admission_source"]

VITAL_FEATURES = [
    f"{prefix}_{stat}"
    for prefix in (*PERIODIC_SIGNALS.values(), *APERIODIC_SIGNALS.values())
    for stat in ["mean", "min", "max", "last"]
]
LAB_FEATURES = [
    f"{lab}_{stat}"
    for lab in LAB_NAMES
    for stat in ["first", "mean", "min", "max", "last"]
]
FEATURES = CLINICAL_FEATURES + VITAL_FEATURES + LAB_FEATURES + ["prediction_hour"]

def read_csv(path, required_columns):
    if not path.is_file():
        raise FileNotFoundError(f"Ficheiro não encontrado: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas em falta em {path.name}: {missing}")
    return df

def find_patient_file():
    for name in ["patient.csv.gz", "patient.csv"]:
        path = DATA_DIR / name
        if path.is_file():
            return path
    for name in ["patient.csv.gz", "patient.csv"]:
        matches = sorted(DATA_DIR.rglob(name)) if DATA_DIR.exists() else []
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Vários ficheiros {name} encontrados em data/")
    raise FileNotFoundError("Não foi encontrado patient.csv.gz nem patient.csv")

def aggregate_vitals(df, signals, cutoff):
    window = df[df["observationoffset"].between(0, cutoff)].copy()
    window = window.sort_values(
        ["patientunitstayid", "observationoffset"], kind="stable"
    )
    results = []
    for original, prefix in signals.items():
        aggregated = (
            window.groupby("patientunitstayid")[original]
            .agg(["mean", "min", "max", "last"])
        )
        aggregated.columns = [
            f"{prefix}_{stat}" for stat in aggregated.columns
        ]
        results.append(aggregated)
    return pd.concat(results, axis=1).reset_index()

def aggregate_labs(df, cutoff):
    window = df[
        df["labresultoffset"].between(0, cutoff)
        & df["labname"].isin(LAB_NAMES)
    ].copy()
    window = window.sort_values(
        ["patientunitstayid", "labresultoffset"], kind="stable"
    )
    if window.empty:
        return pd.DataFrame(columns=["patientunitstayid", *LAB_FEATURES])
    aggregated = (
        window.groupby(["patientunitstayid", "labname"])["labresult"]
        .agg(["first", "mean", "min", "max", "last"])
    )
    wide = aggregated.unstack("labname")
    wide.columns = [f"{lab}_{stat}" for stat, lab in wide.columns]
    return wide.reset_index()

def main():
    patient_path = find_patient_file()
    patients = read_csv(
        patient_path,
        ["patientunitstayid", "uniquepid",
         "unitdischargestatus", "unitdischargeoffset"]
    )
    if patients["patientunitstayid"].duplicated().any():
        raise ValueError("Existem patientunitstayid duplicados em patient.csv")

    patients["unitdischargeoffset"] = pd.to_numeric(
        patients["unitdischargeoffset"], errors="coerce"
    )
    status = (
        patients["unitdischargestatus"]
        .astype("string").str.strip().str.lower()
    )
    patients["target_icu"] = status.map({"alive": 0, "expired": 1})
    patients = patients.dropna(
        subset=["patientunitstayid", "uniquepid",
                "unitdischargeoffset", "target_icu"]
    ).copy()
    patients["target_icu"] = patients["target_icu"].astype(int)

    clinical = read_csv(
        PROCESSED_DIR / "clinical_agent_data.csv",
        ["patientunitstayid", *CLINICAL_FEATURES]
    )[["patientunitstayid", *CLINICAL_FEATURES]].copy()
    if clinical["patientunitstayid"].duplicated().any():
        raise ValueError("Existem episódios duplicados nos dados clínicos")
    clinical["age"] = pd.to_numeric(clinical["age"], errors="coerce")

    periodic = read_csv(
        PROCESSED_DIR / "vitalPeriodic_agent_raw.csv",
        ["patientunitstayid", "observationoffset", *PERIODIC_SIGNALS]
    )[["patientunitstayid", "observationoffset", *PERIODIC_SIGNALS]].copy()

    aperiodic = read_csv(
        PROCESSED_DIR / "vitalAperiodic_agent_raw.csv",
        ["patientunitstayid", "observationoffset", *APERIODIC_SIGNALS]
    )[["patientunitstayid", "observationoffset", *APERIODIC_SIGNALS]].copy()

    labs = read_csv(
        PROCESSED_DIR / "lab_agent_raw.csv",
        ["patientunitstayid", "labresultoffset", "labname", "labresult"]
    )[["patientunitstayid", "labresultoffset", "labname", "labresult"]].copy()

    for df, offset, columns in [
        (periodic, "observationoffset", PERIODIC_SIGNALS),
        (aperiodic, "observationoffset", APERIODIC_SIGNALS),
        (labs, "labresultoffset", ["labresult"])
    ]:
        for col in [offset, *columns]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    all_evaluations = []
    for hour in PREDICTION_HOURS:
        cutoff = hour * 60
        eligible = patients[
            patients["unitdischargeoffset"] > cutoff
        ][["patientunitstayid", "uniquepid", "target_icu"]].copy()
        if eligible.empty:
            continue
        eligible["prediction_hour"] = hour

        periodic_features = aggregate_vitals(periodic, PERIODIC_SIGNALS, cutoff)
        aperiodic_features = aggregate_vitals(
            aperiodic, APERIODIC_SIGNALS, cutoff
        )
        lab_features = aggregate_labs(labs, cutoff)

        for table in [clinical, periodic_features, aperiodic_features, lab_features]:
            eligible = eligible.merge(
                table, on="patientunitstayid",
                how="left", validate="many_to_one"
            )
        all_evaluations.append(eligible)

    if not all_evaluations:
        raise ValueError("Nenhum internamento elegível encontrado")

    dataset = pd.concat(all_evaluations, ignore_index=True)
    for col in VITAL_FEATURES + LAB_FEATURES:
        if col not in dataset.columns:
            dataset[col] = float("nan")
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

    dataset = dataset[
        ["patientunitstayid", "uniquepid", "prediction_hour",
         "target_icu", *[col for col in FEATURES if col != "prediction_hour"]]
    ]
    dataset = dataset.sort_values(
        ["patientunitstayid", "prediction_hour"]
    ).reset_index(drop=True)

    if dataset.duplicated(["patientunitstayid", "prediction_hour"]).any():
        raise ValueError("Existem avaliações duplicadas")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_FILE, index=False)

    summary = dataset.groupby("prediction_hour").agg(
        avaliacoes=("target_icu", "size"),
        mortes=("target_icu", "sum")
    )
    print(f"\nDataset criado: {OUTPUT_FILE}")
    print(
        f"{len(dataset)} avaliações | "
        f"{dataset['patientunitstayid'].nunique()} episódios | "
        f"{dataset['uniquepid'].nunique()} pacientes | "
        f"{len(FEATURES)} features"
    )
    print("\nAvaliações por momento:")
    print(summary.to_string())

if __name__ == "__main__":
    main()
