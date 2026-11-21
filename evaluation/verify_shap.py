
"""
IACH — Verificacao de SHAP e outcome real.

Executar na raiz do projeto:
    python -u evaluation/verify_shap.py

Nao treina nem modifica o modelo.
"""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_model.evaluate_model import load_inputs
from ml_model.predict import (
    original_feature_name,
    predict_mortality,
    CATEGORICAL,
)

EPISODE_ID = "1694608"
OUTPUT = ROOT / "evaluation" / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)


def verify_feature_mapping(pipeline, features):
    """Verifica a correspondencia entre colunas e features."""

    preprocess = pipeline.named_steps["preprocess"]
    names = list(preprocess.get_feature_names_out())

    mapping = {}

    for name in names:
        original = original_feature_name(name)

        if original not in features:
            raise AssertionError(
                f"Feature desconhecida: {name}"
            )

        # As colunas numericas devem corresponder
        # exatamente ao nome original.
        if name.startswith("numeric__"):
            assert name == f"numeric__{original}"

        mapping.setdefault(original, []).append(name)

    assert set(mapping) == set(features), (
        "Nem todas as 53 features estao representadas."
    )

    # Verificar a configuracao real do OneHotEncoder.
    categorical_pipeline = (
        preprocess.named_transformers_["categorical"]
    )
    encoder = categorical_pipeline.named_steps["onehot"]

    expected_encoded = set()

    for feature, categories in zip(
        CATEGORICAL, encoder.categories_
    ):
        for category in categories:
            expected_encoded.add(
                f"categorical__{feature}_{category}"
            )

    actual_encoded = {
        name for name in names
        if name.startswith("categorical__")
    }

    assert actual_encoded == expected_encoded, (
        "As colunas one-hot nao correspondem "
        "as categorias do encoder."
    )

    print(
        f"[OK] {len(features)} features originais "
        f"mapeadas em {len(names)} colunas transformadas."
    )

    return names, mapping


def verify_episode(train, test):
    """Identifica o outcome e confirma o conjunto de teste."""

    in_train = train[
        train["patientunitstayid"].eq(EPISODE_ID)
    ]

    in_test = test[
        test["patientunitstayid"].eq(EPISODE_ID)
    ]

    assert in_train.empty, (
        "O episodio aparece no treino!"
    )

    assert len(in_test) == 1, (
        "O episodio nao aparece uma unica vez no teste."
    )

    row = in_test.iloc[0]
    outcome = int(row["target_icu"])

    assert outcome in (0, 1)

    print(f"[OK] Episodio {EPISODE_ID} pertence ao teste.")
    print(f"Outcome real: {outcome}")

    if outcome == 1:
        print("Resultado observado: MORREU na UCI.")
    else:
        print("Resultado observado: SOBREVIVEU a UCI.")

    return row, outcome


def verify_shap(pipeline, features, row, names, mapping):
    """Compara SHAP direto do XGBoost com predict.py."""

    # Converter os valores para o formato esperado.
    values = {
        feature: (
            None if pd.isna(row[feature])
            else row[feature]
        )
        for feature in features
    }

    # Calculo oficial ja usado no projeto.
    result = predict_mortality(values)

    X = pd.DataFrame([values], columns=features)

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["xgboost"]

    transformed = preprocess.transform(X)

    contributions = model.get_booster().predict(
        xgb.DMatrix(transformed),
        pred_contribs=True
    )

    assert contributions.shape == (1, len(names) + 1)

    grouped = {feature: 0.0 for feature in features}

    # Agregacao independente das contribuicoes.
    for index, transformed_name in enumerate(names):
        original = original_feature_name(transformed_name)
        grouped[original] += float(contributions[0, index])

    direct_shap = {
        item["feature"]: item["shap_value"]
        for item in result["shap_values"]
    }

    for feature in features:
        assert np.isclose(
            grouped[feature],
            direct_shap[feature],
            atol=1e-5
        ), f"Contribuicao SHAP diferente: {feature}"

    print("[OK] SHAP agrupado coincide com predict.py.")

    base = float(contributions[0, -1])
    shap_sum = sum(grouped.values())

    reconstructed = float(expit(base + shap_sum))

    probability = float(
        pipeline.predict_proba(X)[0, 1]
    )

    assert np.isclose(
        reconstructed, probability, atol=1e-5
    )

    assert np.isclose(
        probability,
        result["mortality_probability"],
        atol=1e-5
    )

    print("[OK] SHAP reproduz a previsao do modelo.")

    print(
        f"Probabilidade prevista: {probability:.2%}"
    )
    print(
        f"Probabilidade reconstruida: {reconstructed:.2%}"
    )

    print("\nTop 10 contribuicoes SHAP:")

    for item in result["top_shap_features"]:
        imputed = (
            " (valor em falta/imputado)"
            if item["was_imputed"] else ""
        )

        print(
            f"  {item['feature']}: "
            f"{item['shap_value']:+.4f}{imputed}"
        )

    return result, probability


def main():
    print("=== VERIFICACAO SHAP ===\n")

    train, test, pipeline, features = load_inputs()

    names, mapping = verify_feature_mapping(
        pipeline, features
    )

    row, outcome = verify_episode(train, test)

    result, probability = verify_shap(
        pipeline, features, row, names, mapping
    )

    report = {
        "episode_id": EPISODE_ID,
        "dataset_split": "test",
        "observed_outcome": outcome,
        "observed_outcome_label": (
            "died_in_icu" if outcome == 1
            else "survived_icu"
        ),
        "predicted_mortality_probability": probability,
        "n_original_features": len(features),
        "n_transformed_features": len(names),
        "n_missing_features": result["n_missing_features"],
        "feature_mapping_verified": True,
        "shap_agreement_verified": True,
        "shap_reconstruction_verified": True,
        "top_shap_features": result["top_shap_features"]
    }

    output_path = OUTPUT / "shap_verification_24h.json"

    with output_path.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            report, f, indent=2,
            ensure_ascii=False
        )

    print("\n=== VERIFICACAO CONCLUIDA ===")
    print(f"Relatorio guardado em: {output_path}")


if __name__ == "__main__":
    main()
