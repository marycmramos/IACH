"""IACH — avaliação exploratória exclusivamente 24h.

Executar: python -u ml_model/evaluate_model.py
Lê o modelo treinado e a divisão guardada por train_model.py.
O limiar experimental é selecionado por previsões out-of-fold do treino.
Nunca usa o teste para escolher um limiar ou calibrar probabilidades.
"""
from pathlib import Path
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ml_model" / "ml_features_24h.csv"
MODEL = ROOT / "ml_model" / "models" / "xgboost_24h.joblib"
SPLIT = ROOT / "ml_model" / "results" / "split_24h.json"
OUTPUT = ROOT / "ml_model" / "results"
TARGET = "target_icu"
N_BOOTSTRAP = 1000
SEED = 42
MIN_SENSITIVITY = 0.70
THRESHOLDS = np.round(np.arange(0.005, 0.301, 0.005), 4)


def load_inputs():
    for path in (DATA, MODEL, SPLIT):
        if not path.is_file():
            raise FileNotFoundError(f"Falta {path}; executa prepare_data.py e train_model.py.")
    df = pd.read_csv(DATA, dtype={"uniquepid": "string", "patientunitstayid": "string"})
    artifact = joblib.load(MODEL)
    with SPLIT.open(encoding="utf-8") as f:
        split = json.load(f)
    if artifact.get("mode") != "24h" or len(artifact["features"]) != 53:
        raise ValueError("Modelo incompatível com o dataset de 24h.")
    features = artifact["features"]
    if not {"uniquepid", "patientunitstayid", TARGET, *features}.issubset(df.columns):
        raise ValueError("Dataset incompleto.")
    for col in features:
        if col in ("gender", "unit_type", "admission_source"):
            df[col] = df[col].astype("string").str.strip().replace("", pd.NA).astype(object)
            df[col] = df[col].where(df[col].notna(), np.nan)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    tr_ids, te_ids = set(split["train_patient_ids"]), set(split["test_patient_ids"])
    if tr_ids & te_ids or set(df.uniquepid.astype(str)) != tr_ids | te_ids:
        raise ValueError("A divisão de pacientes não corresponde ao dataset. Volta a treinar.")
    train = df[df.uniquepid.isin(tr_ids)].copy()
    test = df[df.uniquepid.isin(te_ids)].copy()
    if len(train) != split["n_train"] or len(test) != split["n_test"]:
        raise ValueError("O número de episódios mudou; volta a treinar.")
    return train, test, artifact["pipeline"], features


def metrics(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    return {"n": int(len(y)), "deaths": int(y.sum()), "prevalence": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "mean_predicted_probability": float(p.mean())}


def bootstrap(test):
    groups = [g for _, g in test.groupby("uniquepid", sort=False)]
    rng = np.random.default_rng(SEED)
    scores = {key: [] for key in ("roc_auc", "pr_auc", "brier")}
    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(0, len(groups), size=len(groups))
        sample = pd.concat([groups[i] for i in indices], ignore_index=True)
        if sample[TARGET].nunique() != 2:
            continue
        s = metrics(sample[TARGET], sample["probability"])
        for key in scores:
            scores[key].append(s[key])
    if min(map(len, scores.values())) < 100:
        raise ValueError("Poucas reamostragens bootstrap válidas.")
    return {key: {"lower_95": float(np.percentile(v, 2.5)),
                  "upper_95": float(np.percentile(v, 97.5))} for key, v in scores.items()}


def threshold_metrics(y, p, t):
    tn, fp, fn, tp = confusion_matrix(
        y, np.asarray(p) >= t, labels=[0, 1]
    ).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return {
        "threshold": float(t),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "sensitivity": float(recall),
        "specificity": float(specificity)
    }

def patient_weights(df):
    # Mesma ponderação do treino, com class_factor=1.
    repetitions = df.groupby("uniquepid")["uniquepid"].transform("size").to_numpy(dtype=float)
    w = 1.0 / repetitions
    return w * len(w) / w.sum()


def threshold_cv(train, pipeline, features):
    patients = train.groupby("uniquepid", sort=True)[TARGET].max().reset_index()
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    oof = pd.Series(index=train.index, dtype=float)
    for a, b in folds.split(patients.uniquepid, patients[TARGET]):
        tr_ids = set(patients.iloc[a].uniquepid)
        va_ids = set(patients.iloc[b].uniquepid)
        tr = train[train.uniquepid.isin(tr_ids)]
        va = train[train.uniquepid.isin(va_ids)]
        model = clone(pipeline)
        model.fit(tr[features], tr[TARGET], xgboost__sample_weight=patient_weights(tr))
        oof.loc[va.index] = model.predict_proba(va[features])[:, 1]
    if oof.isna().any():
        raise ValueError("Previsões OOF incompletas.")
    rows = [threshold_metrics(train[TARGET], oof.loc[train.index], t) for t in THRESHOLDS]
    grid = pd.DataFrame(rows)
    eligible = grid[grid.sensitivity >= MIN_SENSITIVITY]
    if eligible.empty:
        return None, grid, "Nenhum limiar atingiu o objetivo de sensibilidade."
    chosen = eligible.sort_values(["specificity", "precision", "threshold"], ascending=False).iloc[0]
    return float(chosen.threshold), grid, "Limiar experimental selecionado só na validação cruzada."


def plots(test):
    bins = test.copy()
    bins["bin"] = pd.qcut(bins.probability, q=5, duplicates="drop")
    summary = bins.groupby("bin", observed=True).agg(
        n=(TARGET, "size"), deaths=(TARGET, "sum"), predicted=("probability", "mean"), observed=(TARGET, "mean")
    ).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", label="Ideal")
    ax.plot(summary.predicted, summary.observed, "o-", label="Observado por quantis")
    lim = min(1.0, max(.12, float(summary[["predicted", "observed"]].max().max()) * 1.2))
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Probabilidade prevista", ylabel="Frequência observada",
           title="Calibração descritiva — 24h")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "calibration_24h.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, name in ((0, "Sobreviveu"), (1, "Morreu")):
        values = test.loc[test[TARGET] == label, "probability"]
        ax.hist(values, bins=np.linspace(0, 1, 31), alpha=0.6, label=f"{name} (n={len(values)})")
    ax.set(xlabel="Probabilidade prevista", ylabel="N.º de episódios", title="Distribuição de risco — 24h")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "probability_distribution_24h.png", dpi=150)
    plt.close(fig)
    return summary


def main():
    train, test, pipeline, features = load_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    test["probability"] = pipeline.predict_proba(test[features])[:, 1]
    overall = metrics(test[TARGET], test.probability)
    ci = bootstrap(test)
    threshold, grid, note = threshold_cv(train, pipeline, features)
    bins = plots(test)
    bins.to_csv(OUTPUT / "calibration_bins_24h.csv", index=False)
    grid.to_csv(OUTPUT / "threshold_grid_24h.csv", index=False)
    threshold_result = threshold_metrics(test[TARGET], test.probability, threshold) if threshold is not None else None
    with (OUTPUT / "evaluation_24h.json").open("w", encoding="utf-8") as f:
        json.dump({"metrics": overall, "bootstrap_95": ci,
                   "threshold_experiment": {"chosen_on_training_oof": threshold,
                       "target_sensitivity": MIN_SENSITIVITY, "test_descriptive": threshold_result, "note": note},
                   "warning": "Teste exploratório; nenhuma calibração nem otimização foi feita no teste."},
                  f, indent=2, ensure_ascii=False)
    print(f"Teste: {overall['n']} episódios, {overall['deaths']} mortes")
    print(f"ROC-AUC: {overall['roc_auc']:.4f} | IC95 {ci['roc_auc']['lower_95']:.4f}–{ci['roc_auc']['upper_95']:.4f}")
    print(f"PR-AUC:  {overall['pr_auc']:.4f} | IC95 {ci['pr_auc']['lower_95']:.4f}–{ci['pr_auc']['upper_95']:.4f}")
    print(f"Brier:   {overall['brier']:.4f}")
    print(f"Threshold experimental OOF: {threshold}; {note}")
    print(f"Resultados guardados em: {OUTPUT}")


if __name__ == "__main__":
    main()
