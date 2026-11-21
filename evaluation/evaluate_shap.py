
"""
IACH — Avaliação SHAP do modelo XGBoost de 24h.

Gera:
- Importância global das variáveis (SHAP)
- Waterfall de uma previsão individual
- CSV com importância média das variáveis
- JSON com informação sobre a previsão analisada

Executar na raiz do projeto:
    python -u evaluation/evaluate_shap.py
"""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
from scipy.special import expit

# Caminhos do projeto
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_model.evaluate_model import load_inputs
from ml_model.predict import original_feature_name

OUTPUT = ROOT / "evaluation" / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)

TOP_N = 15


def calculate_shap(pipeline, X, features):
    """Calcula TreeSHAP para as 53 features originais."""

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["xgboost"]

    # Aplicar o preprocessing já ajustado no treino
    transformed = preprocess.transform(X)
    transformed_names = preprocess.get_feature_names_out()

    # Contribuições SHAP em log-odds
    contributions = model.get_booster().predict(
        xgb.DMatrix(transformed),
        pred_contribs=True
    )

    shap_encoded = contributions[:, :-1]
    base_values = contributions[:, -1]

    # Agrupar as colunas one-hot nas features originais
    shap_grouped = np.zeros((len(X), len(features)))

    feature_indices = {
        name: i for i, name in enumerate(features)
    }

    for j, transformed_name in enumerate(transformed_names):
        original = original_feature_name(transformed_name)
        i = feature_indices[original]

        shap_grouped[:, i] += shap_encoded[:, j]

    # Confirmar que os valores SHAP reproduzem
    # as probabilidades previstas pelo modelo
    reconstructed = expit(
        base_values + shap_grouped.sum(axis=1)
    )

    predicted = pipeline.predict_proba(X)[:, 1]

    if not np.allclose(
        reconstructed, predicted, atol=1e-5
    ):
        raise ValueError(
            "Os valores SHAP não reproduzem as previsões."
        )

    return shap_grouped, base_values, predicted


def plot_global(shap_values, features):
    """Gráfico das contribuições SHAP médias absolutas."""

    importance = np.abs(shap_values).mean(axis=0)

    df = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": importance
    }).sort_values(
        "mean_abs_shap", ascending=False
    )

    df.to_csv(
        OUTPUT / "shap_feature_importance_24h.csv",
        index=False
    )

    top = df.head(TOP_N).iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(
        top["feature"],
        top["mean_abs_shap"]
    )

    plt.xlabel("Mean absolute SHAP value (log-odds)")
    plt.ylabel("Clinical feature")
    plt.title(
        "Global SHAP Feature Importance — ICU Mortality"
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT / "shap_global_24h.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\nTop 10 features:")
    print(df.head(10).to_string(index=False))


def plot_local(
    shap_values, base_values, predicted,
    X, features
):
    """Waterfall para o episódio de maior risco previsto."""

    # Escolher um caso ilustrativo sem consultar
    # o outcome real do paciente
    idx = int(np.argmax(predicted))

    row = X.iloc[idx]

    # Valores originais, antes do preprocessing
    original_values = row[features].to_numpy(
        dtype=object
    )

    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=float(base_values[idx]),
        data=original_values,
        feature_names=features
    )

    shap.plots.waterfall(
        explanation,
        max_display=12,
        show=False
    )

    plt.title(
        "Local SHAP Explanation — ICU Mortality",
        pad=20
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT / "shap_waterfall_24h.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    result = {
        "selection": "highest predicted risk in test set",
        "patientunitstayid": str(
            row["patientunitstayid"]
        ),
        "predicted_probability": float(predicted[idx]),
        "base_log_odds": float(base_values[idx]),
        "shap_unit": "log-odds",
        "top_features": []
    }

    ordered = np.argsort(
        np.abs(shap_values[idx])
    )[::-1]

    for j in ordered[:10]:
        raw = row[features[j]]

        result["top_features"].append({
            "feature": features[j],
            "value": (
                None if pd.isna(raw)
                else str(raw)
            ),
            "shap_value": float(shap_values[idx, j])
        })

    with (
        OUTPUT / "shap_local_24h.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            result, f,
            indent=2, ensure_ascii=False
        )

    print(
        "\nEpisódio analisado:",
        result["patientunitstayid"]
    )
    print(
        "Risco previsto:",
        f"{result['predicted_probability']:.2%}"
    )


def main():
    print("A carregar modelo e dados de teste...")

    train, test, pipeline, features = load_inputs()

    print(f"Episódios de teste: {len(test)}")
    print(f"Features originais: {len(features)}")

    X = test[features]

    print("\nA calcular SHAP...")

    shap_values, base_values, predicted = (
        calculate_shap(pipeline, X, features)
    )

    print("A gerar importância global...")
    plot_global(shap_values, features)

    print("\nA gerar waterfall individual...")
    plot_local(
        shap_values, base_values,
        predicted, test, features
    )

    print("\nAvaliação SHAP concluída!")
    print(f"Resultados guardados em: {OUTPUT}")


if __name__ == "__main__":
    main()
