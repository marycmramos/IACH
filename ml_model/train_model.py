
"""IACH — treino XGBoost para previsão de mortalidade na UCI às 24h.
Executar: python -u ml_model/train_model.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "ml_model" / "ml_features_24h.csv"
MODEL = ROOT / "ml_model" / "models" / "xgboost_24h.joblib"
RESULTS = ROOT / "ml_model" / "results"
TARGET = "target_icu"
CATEGORICAL = ("gender", "unit_type", "admission_source")
CLINICAL = ("age", "gender", "unit_type", "admission_source")
VITAL = tuple(f"{s}_{stat}" for s in ("hr", "sao2", "rr", "sbp", "dbp", "map") for stat in ("mean", "min", "max", "last"))
LAB = tuple(f"{s}_{stat}" for s in ("BUN", "creatinine", "glucose", "potassium", "sodium") for stat in ("first", "mean", "min", "max", "last"))
FEATURES = list(CLINICAL + VITAL + LAB)
PARAMS = {"n_estimators": 220, "max_depth": 2, "learning_rate": 0.03, "class_factor": 1.0}
SEED = 42

def load_data():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Falta {INPUT}; executa prepare_data.py primeiro.")
    df = pd.read_csv(INPUT, dtype={"uniquepid": "string", "patientunitstayid": "string"}, low_memory=False)
    expected = {"uniquepid", "patientunitstayid", TARGET, *FEATURES}
    if set(df.columns) != expected:
        raise ValueError(f"Colunas em falta: {sorted(expected - set(df.columns))}; extra: {sorted(set(df.columns) - expected)}")
    if df[["uniquepid", "patientunitstayid", TARGET]].isna().any().any():
        raise ValueError("IDs ou target em falta.")
    if df["patientunitstayid"].duplicated().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError("Episódios duplicados ou target inválido.")
    df[TARGET] = df[TARGET].astype(int)
    for col in FEATURES:
        if col in CATEGORICAL:
            df[col] = df[col].astype("string").str.strip().replace("", pd.NA).astype(object)
            df[col] = df[col].where(df[col].notna(), np.nan)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df.sort_values("patientunitstayid").reset_index(drop=True)

def make_pipeline():
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), [x for x in FEATURES if x not in CATEGORICAL]),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), list(CATEGORICAL))
    ])
    xgb = XGBClassifier(n_estimators=PARAMS["n_estimators"], max_depth=PARAMS["max_depth"],
        learning_rate=PARAMS["learning_rate"], subsample=0.9, colsample_bytree=0.9,
        min_child_weight=3, objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=42, n_jobs=4)
    return Pipeline([("preprocess", preprocess), ("xgboost", xgb)])

def patient_weights(df):
    repeats = df.groupby("uniquepid")["uniquepid"].transform("size").to_numpy(dtype=float)
    weights = (1.0 / repeats) * np.where(df[TARGET].to_numpy() == 1, PARAMS["class_factor"], 1.0)
    return weights * len(weights) / weights.sum()

def split_data(df):
    for seed in range(SEED, SEED + 100):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        a, b = next(splitter.split(df, groups=df["uniquepid"]))
        train, test = df.iloc[a].copy(), df.iloc[b].copy()
        if train[TARGET].nunique() == 2 and test[TARGET].nunique() == 2:
            return train, test, seed
    raise ValueError("Não foi possível dividir os pacientes com ambas as classes.")

def patient_folds(train):
    patients = train.groupby("uniquepid", sort=True)[TARGET].max().reset_index()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    for a, b in cv.split(patients["uniquepid"], patients[TARGET]):
        train_ids = set(patients.iloc[a]["uniquepid"])
        val_ids = set(patients.iloc[b]["uniquepid"])
        tr = train[train["uniquepid"].isin(train_ids)]
        va = train[train["uniquepid"].isin(val_ids)]
        if tr[TARGET].nunique() != 2 or va[TARGET].nunique() != 2:
            raise ValueError("Fold sem ambas as classes.")
        yield tr, va

def score(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    return {"n": int(len(y)), "deaths": int(y.sum()), "prevalence": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "mean_probability": float(p.mean())}

def main():
    df = load_data()
    train, test, seed = split_data(df)
    if not set(train["uniquepid"]).isdisjoint(set(test["uniquepid"])):
        raise ValueError("Data leakage: pacientes nos dois conjuntos.")
    print(f"Dataset 24h: {len(df)} episódios | {df['uniquepid'].nunique()} pacientes | {len(FEATURES)} features")
    print(f"Treino: {len(train)} episódios; teste: {len(test)} episódios; seed={seed}")
    cv_results = []
    for index, (tr, va) in enumerate(patient_folds(train), 1):
        model = make_pipeline()
        model.fit(tr[FEATURES], tr[TARGET], xgboost__sample_weight=patient_weights(tr))
        result = score(va[TARGET], model.predict_proba(va[FEATURES])[:, 1])
        cv_results.append(result)
        print(f"CV {index}/3: ROC-AUC={result['roc_auc']:.3f}; PR-AUC={result['pr_auc']:.3f}", flush=True)
    model = make_pipeline()
    model.fit(train[FEATURES], train[TARGET], xgboost__sample_weight=patient_weights(train))
    test_score = score(test[TARGET], model.predict_proba(test[FEATURES])[:, 1])
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "split_24h.json").open("w", encoding="utf-8") as f:
        json.dump({"seed": seed, "train_patient_ids": sorted(train["uniquepid"].astype(str).unique().tolist()),
                   "test_patient_ids": sorted(test["uniquepid"].astype(str).unique().tolist()),
                   "n_train": len(train), "n_test": len(test)}, f, indent=2)
    joblib.dump({"pipeline": model, "features": FEATURES, "mode": "24h"}, MODEL)
    with (RESULTS / "training_24h.json").open("w", encoding="utf-8") as f:
        json.dump({"parameters": PARAMS, "cv_folds": cv_results, "test": test_score,
                   "warning": "Teste exploratório; modelo não validado para utilização clínica."}, f, indent=2)
    print(f"Teste: ROC-AUC={test_score['roc_auc']:.4f}; PR-AUC={test_score['pr_auc']:.4f}; Brier={test_score['brier']:.4f}")
    print(f"Modelo guardado: {MODEL}")

if __name__ == "__main__":
    main()
