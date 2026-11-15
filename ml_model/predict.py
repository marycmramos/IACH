
"""IACH — inferência de mortalidade na UCI com XGBoost + SHAP (24h).

Utilização pelo Severity Agent:
    from ml_model.predict import predict_mortality
    result = predict_mortality(features)

Teste com um episódio reservado:
    python -u ml_model/predict.py 156906

A função predict_mortality não consulta CSVs nem utiliza o target real.
"""
from pathlib import Path
from functools import lru_cache
import json
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "ml_model" / "models" / "xgboost_24h.joblib"
DATA_PATH = ROOT / "ml_model" / "ml_features_24h.csv"

CATEGORICAL = ("gender", "unit_type", "admission_source")
TOP_N = 10

@lru_cache(maxsize=1)
def load_model():
    """Carrega o modelo uma única vez e reutiliza-o."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict):
        raise ValueError("Formato do modelo inválido.")
    if artifact.get("mode") != "24h":
        raise ValueError("Este módulo apenas suporta o modelo de 24h.")

    pipeline = artifact["pipeline"]
    features = list(artifact["features"])

    if len(features) != 53 or len(set(features)) != len(features):
        raise ValueError("Lista de features inesperada no modelo.")
    if "prediction_hour" in features:
        raise ValueError("O modelo de 24h não deve incluir prediction_hour.")
    if not set(CATEGORICAL).issubset(features):
        raise ValueError("Faltam features categóricas no modelo.")

    return pipeline, features

def prepare_features(values, expected_features):
    """Valida e organiza as features na ordem utilizada no treino."""
    if not isinstance(values, dict):
        raise TypeError("As features devem ser fornecidas num dicionário.")

    missing_keys = sorted(set(expected_features) - set(values))
    extra_keys = sorted(set(values) - set(expected_features))

    if missing_keys:
        raise ValueError(
            f"Features obrigatórias não fornecidas: {missing_keys}. "
            "Se não houver medição, fornece None."
        )
    if extra_keys:
        raise ValueError(f"Features desconhecidas: {extra_keys}")

    cleaned = {}
    missing_features = []

    for name in expected_features:
        value = values[name]

        if value is None or (not isinstance(value, (list, dict, tuple))
                             and pd.isna(value)):
            cleaned[name] = np.nan
            missing_features.append(name)
            continue

        if name in CATEGORICAL:
            if not isinstance(value, str):
                raise ValueError(f"{name} deve ser uma string ou None.")
            value = value.strip()
            if not value:
                cleaned[name] = np.nan
                missing_features.append(name)
            else:
                cleaned[name] = value
        else:
            if isinstance(value, bool):
                raise ValueError(f"{name} não pode ser booleano.")
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Valor numérico inválido em {name}: {value}"
                ) from None

            if not np.isfinite(number):
                raise ValueError(f"Valor não finito em {name}.")
            cleaned[name] = number

    X = pd.DataFrame([cleaned], columns=expected_features)

    for name in CATEGORICAL:
        X[name] = X[name].astype(object).where(X[name].notna(), np.nan)

    return X, missing_features

def original_feature_name(transformed_name):
    """Relaciona uma coluna transformada com a sua feature original."""
    name = str(transformed_name)

    for prefix in ("numeric__", "num__"):
        if name.startswith(prefix):
            return name.removeprefix(prefix)

    for prefix in ("categorical__", "cat__"):
        if name.startswith(prefix):
            encoded = name.removeprefix(prefix)
            for feature in sorted(CATEGORICAL, key=len, reverse=True):
                if encoded.startswith(feature + "_"):
                    return feature

    raise ValueError(f"Feature transformada desconhecida: {name}")

def explain_prediction(pipeline, X, features, missing_features):
    """Calcula TreeSHAP e recupera os valores usados pelo modelo."""
    preprocess = pipeline.named_steps["preprocess"]
    fitted_model = pipeline.named_steps["xgboost"]

    transformed = preprocess.transform(X)
    transformed_names = list(preprocess.get_feature_names_out())

    if hasattr(transformed, "toarray"):
        transformed_array = transformed.toarray()
    else:
        transformed_array = np.asarray(transformed)

    if transformed_array.shape[0] != 1:
        raise ValueError("Era esperada uma única avaliação.")

    contributions = fitted_model.get_booster().predict(
        xgb.DMatrix(transformed),
        pred_contribs=True
    )

    if contributions.shape != (1, len(transformed_names) + 1):
        raise ValueError("Dimensão inesperada das contribuições SHAP.")

    shap_row = contributions[0, :-1]
    base_log_odds = float(contributions[0, -1])

    grouped_shap = {name: 0.0 for name in features}
    processed_numeric = {}

    for index, transformed_name in enumerate(transformed_names):
        original = original_feature_name(transformed_name)

        if original not in grouped_shap:
            raise ValueError(f"Feature inesperada: {original}")

        grouped_shap[original] += float(shap_row[index])

        if original not in CATEGORICAL:
            processed_numeric[original] = float(transformed_array[0, index])

    shap_values = []

    for name, contribution in grouped_shap.items():
        raw_value = X.iloc[0][name]
        was_imputed = name in missing_features

        if pd.isna(raw_value):
            original_value = None
        elif name in CATEGORICAL:
            original_value = str(raw_value)
        else:
            original_value = float(raw_value)

        shap_values.append({
            "feature": name,
            "original_value": original_value,
            "was_imputed": was_imputed,
            "processed_value": (
                None if name in CATEGORICAL
                else processed_numeric.get(name)
            ),
            "shap_value": float(contribution),
            "direction": (
                "aumenta" if contribution > 0
                else "diminui" if contribution < 0
                else "neutro"
            )
        })

    shap_values.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True
    )

    reconstructed_probability = float(
        expit(base_log_odds + sum(grouped_shap.values()))
    )

    return {
        "base_log_odds": base_log_odds,
        "base_probability": float(expit(base_log_odds)),
        "reconstructed_probability": reconstructed_probability,
        "shap_values": shap_values
    }

def predict_mortality(features):
    """Executa XGBoost + SHAP a partir de 53 features das primeiras 24h.

    Não recebe patientunitstayid, target, dados de alta ou APACHE.
    Não consulta o dataset e não guarda ficheiros.
    """
    pipeline, expected_features = load_model()

    X, missing_features = prepare_features(features, expected_features)

    probability = float(pipeline.predict_proba(X)[0, 1])

    explanation = explain_prediction(
        pipeline, X, expected_features, missing_features
    )

    if not np.isclose(
        probability,
        explanation["reconstructed_probability"],
        atol=1e-5,
        rtol=1e-5
    ):
        raise ValueError(
            "As contribuições SHAP não reproduzem a previsão do modelo."
        )

    return {
        "prediction_hour": 24,
        "outcome": "icu_mortality_after_24h",
        "mortality_probability": probability,
        "base_probability": explanation["base_probability"],
        "shap_unit": "log_odds",
        "n_features": len(expected_features),
        "n_missing_features": len(missing_features),
        "missing_features": missing_features,
        "top_shap_features": explanation["shap_values"][:TOP_N],
        "shap_values": explanation["shap_values"],
        "warnings": [
            "Modelo académico não validado para utilização clínica.",
            "As probabilidades não foram formalmente calibradas.",
            "SHAP explica o modelo e não demonstra causalidade.",
            "Features imputadas não representam medições reais."
        ]
    }

def test_with_episode(episode_id):
    """Demonstração local com um episódio do dataset de 24h."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {DATA_PATH}")
    data = pd.read_csv(
        DATA_PATH,
        dtype={"uniquepid": "string", "patientunitstayid": "string"},
        low_memory=False
    )
    episode = data[data["patientunitstayid"].eq(str(episode_id))]
    if len(episode) != 1:
        raise ValueError("Episódio não encontrado uma única vez no dataset de 24h.")

    _, expected_features = load_model()
    row = episode.iloc[0]
    features = {
        name: None if pd.isna(row[name]) else row[name]
        for name in expected_features
    }
    result = predict_mortality(features)

    print(f"Episódio: {episode_id} | avaliação: 24h")
    print(f"Probabilidade de mortalidade: {result['mortality_probability']:.2%}")
    print(f"Referência SHAP: {result['base_probability']:.2%}")
    print(f"Features em falta: {result['n_missing_features']}")
    print("\nTop 10 contribuições SHAP:")
    for item in result["top_shap_features"]:
        label = item["feature"]
        if item["was_imputed"]:
            label += " (imputado)"
        print(f"  {label}: {item['shap_value']:+.4f}")
    print("\nTeste de inferência e SHAP concluído.")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python -u ml_model/predict.py [patientunitstayid]")
        sys.exit(1)

    test_with_episode(sys.argv[1])
