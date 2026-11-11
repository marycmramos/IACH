"""IACH — treino e avaliação inicial de XGBoost para mortalidade na UCI.

Entrada: ml_model/ml_features_dynamic.csv (gerado por prepare_data.py).
Separação por uniquepid; nenhuma observação de teste é usada no treino.
Avaliações: 12h, 16h, 20h, 24h. O target é morte posterior na UCI.
Uso académico: estimativas de risco não estão calibradas/validadas clinicamente.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "ml_model" / "ml_features_dynamic.csv"
OUTPUT = ROOT / "ml_model" / "saved_models"
HOURS = (12, 16, 20, 24)
ID_COLUMNS = ("patientunitstayid", "uniquepid")
TARGET = "target_icu"
CLINICAL = ("age", "gender", "unit_type", "admission_source")
CATEGORICAL = ("gender", "unit_type", "admission_source")
VITAL = tuple(
    f"{name}_{stat}"
    for name in ("hr", "sao2", "rr", "sbp", "dbp", "map")
    for stat in ("mean", "min", "max", "last")
)
LAB = tuple(
    f"{name}_{stat}"
    for name in ("BUN", "creatinine", "glucose", "potassium", "sodium")
    for stat in ("first", "mean", "min", "max", "last")
)
FEATURES = (*CLINICAL, *VITAL, *LAB, "prediction_hour")
NUMERIC = tuple(x for x in FEATURES if x not in CATEGORICAL)


def validate_data(df):
    required = set((*ID_COLUMNS, TARGET, *FEATURES))
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias em falta: {missing}")
    extra = sorted(set(df.columns) - required)
    if extra:
        raise ValueError(f"Colunas inesperadas (possível leakage): {extra}")
    if df[list(ID_COLUMNS)].isna().any().any():
        raise ValueError("Existem identificadores em falta.")
    if df.duplicated(["patientunitstayid", "prediction_hour"]).any():
        raise ValueError("Existem avaliações repetidas para o mesmo episódio e hora.")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError("target_icu tem de conter apenas 0 e 1, sem nulos.")
    df[TARGET] = df[TARGET].astype(int)
    if df["prediction_hour"].isna().any() or not df["prediction_hour"].isin(HOURS).all():
        raise ValueError("prediction_hour deve conter apenas 12, 16, 20 ou 24.")
    if (df.groupby("patientunitstayid")[TARGET].nunique() > 1).any():
        raise ValueError("Target inconsistente entre avaliações do mesmo episódio.")
    if df[list(NUMERIC)].replace([np.inf, -np.inf], np.nan).isna().all(axis=0).any():
        empty = df[list(NUMERIC)].replace([np.inf, -np.inf], np.nan).columns[
            df[list(NUMERIC)].replace([np.inf, -np.inf], np.nan).isna().all(axis=0)
        ].tolist()
        raise ValueError(f"Features numéricas totalmente vazias: {empty}")
    df[list(NUMERIC)] = df[list(NUMERIC)].replace([np.inf, -np.inf], np.nan)
    return df


def audit(df):
    print(f"Dataset: {len(df)} avaliações | "
          f"{df['patientunitstayid'].nunique()} episódios | "
          f"{df['uniquepid'].nunique()} pacientes")
    print("Avaliações e mortes por hora:")
    print(df.groupby("prediction_hour")[TARGET].agg(["count", "sum"]).to_string())
    missing = df[list(FEATURES)].isna().mean().mul(100).sort_values(ascending=False)
    print("\nTop 8 features com mais valores em falta (%):")
    print(missing.head(8).round(1).to_string())
    # Intervalos amplos apenas para sinalizar possíveis erros de unidades/medição.
    limits = {
        "age": (0, 120), "hr": (20, 260), "sao2": (0, 100),
        "rr": (0, 100), "sbp": (20, 300), "dbp": (10, 200),
        "map": (10, 230), "creatinine": (0, 30),
        "glucose": (10, 1500), "potassium": (1, 12),
        "sodium": (80, 200), "BUN": (0, 250),
    }
    suspicious = {}
    for col in NUMERIC:
        prefix = next((p for p in limits if col == p or col.startswith(p + "_")), None)
        if prefix is None:
            continue
        lo, hi = limits[prefix]
        count = int((df[col].notna() & ~df[col].between(lo, hi)).sum())
        if count:
            suspicious[col] = count
    print("\nPossíveis valores extremos (apenas alerta; nada é removido):")
    print(suspicious if suspicious else "Nenhum segundo os limites de triagem.")
    print("Nota: o CSV agregado não permite confirmar os offsets das medições; "
          "isso deve ser verificado em prepare_data.py.")
    return {
        "missing_percent": {k: round(float(v), 2) for k, v in missing.items()},
        "extreme_values_counts": suspicious,
    }


def metrics(y, probability):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(y) == 0:
        return None
    result = {
        "n": int(len(y)), "deaths": int(y.sum()),
        "prevalence": round(float(y.mean()), 5),
        "roc_auc": None, "pr_auc": None,
        "brier_score": round(float(brier_score_loss(y, probability)), 5),
    }
    if len(np.unique(y)) == 2:
        result["roc_auc"] = round(float(roc_auc_score(y, probability)), 5)
        result["pr_auc"] = round(float(average_precision_score(y, probability)), 5)
    # 0.5 é um limiar apenas ilustrativo, não otimizado para decisão clínica.
    pred = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result.update({
        "threshold": 0.5, "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
        "sensitivity": round(float(recall_score(y, pred, zero_division=0)), 5),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 5),
    })
    return result


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Não existe {INPUT}. Executa primeiro prepare_data.py")
    df = pd.read_csv(INPUT, dtype={"uniquepid": "string"}, low_memory=False)
    df = validate_data(df)
    diagnostics = audit(df)

    # Split por paciente, NUNCA por linha/episódio.
    # Escolhe a primeira divisão (sem consultar métricas de desempenho)
    # em que treino e teste contêm ambas as classes.
    for seed in range(42, 142):
        split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_idx, test_idx = next(split.split(df, df[TARGET], groups=df["uniquepid"]))
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        if train[TARGET].nunique() == 2 and test[TARGET].nunique() == 2:
            break
    else:
        raise ValueError("Não foi possível criar treino/teste com ambas as classes.")
    assert set(train["uniquepid"]).isdisjoint(set(test["uniquepid"]))
    print(f"\nTreino: {len(train)} avaliações, {train['uniquepid'].nunique()} pacientes, "
          f"{train[TARGET].sum()} registos positivos")
    print(f"Teste: {len(test)} avaliações, {test['uniquepid'].nunique()} pacientes, "
          f"{test[TARGET].sum()} registos positivos")

    # Cada paciente recebe peso total 1, repartido pelas suas avaliações.
    counts = train.groupby("uniquepid")["uniquepid"].transform("size")
    weights = 1.0 / counts.to_numpy(dtype=float)
    y_train = train[TARGET].to_numpy(dtype=int)
    positive_weight = float(weights[y_train == 1].sum())
    negative_weight = float(weights[y_train == 0].sum())
    if positive_weight <= 0:
        raise ValueError("Não há registos positivos no treino.")
    class_ratio = negative_weight / positive_weight

    transformer = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", keep_empty_features=True), list(NUMERIC)),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), list(CATEGORICAL)),
    ])
    model = Pipeline([
        ("preprocess", transformer),
        ("xgboost", XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=class_ratio, tree_method="hist",
            random_state=42, n_jobs=4,
        )),
    ])
    model.fit(train[list(FEATURES)], y_train, xgboost__sample_weight=weights)
    probabilities = model.predict_proba(test[list(FEATURES)])[:, 1]

    results = {
        "definition": "Mortalidade posterior na UCI, previsão às 12/16/20/24h",
        "split": f"80/20 por uniquepid, seed={seed}",
        "features": list(FEATURES),
        "weighting": "peso total igual por paciente; scale_pos_weight no treino",
        "note": "Probabilidades não calibradas; teste usado apenas para avaliação.",
        "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
        "diagnostics": diagnostics,
        "overall": metrics(test[TARGET], probabilities),
        "by_hour": {},
    }
    predictions = test[["patientunitstayid", "uniquepid", "prediction_hour", TARGET]].copy()
    predictions["mortality_probability"] = probabilities
    for hour in HOURS:
        rows = predictions[predictions["prediction_hour"] == hour]
        results["by_hour"][str(hour)] = metrics(rows[TARGET], rows["mortality_probability"])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT / "xgboost_mortality.joblib")
    predictions.to_csv(OUTPUT / "test_predictions.csv", index=False)
    # Permite à futura demonstração selecionar apenas episódios do conjunto de teste.
    test[["patientunitstayid", "uniquepid", "prediction_hour"]].to_csv(
        OUTPUT / "test_patient_ids.csv", index=False
    )
    with (OUTPUT / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nAvaliação global (teste):", results["overall"])
    print("\nAvaliação por momento (teste):")
    for hour, value in results["by_hour"].items():
        print(f"{hour}h: {value}")
    print(f"\nModelo e resultados guardados em: {OUTPUT}")
    print("ATENÇÃO: as probabilidades ainda não estão calibradas para uso clínico.")


if __name__ == "__main__":
    main()
