import pandas as pd
from agents.severity.severity_agent import run_severity_agent
from ml_model.predict import load_model

PATIENT_ID = 156906

df = pd.read_csv("ml_model/ml_features_24h.csv")
row = df.loc[df["patientunitstayid"] == PATIENT_ID].iloc[0]

_, feature_names = load_model()
features = {
    name: None if pd.isna(row[name]) else row[name]
    for name in feature_names
}

severity_input = {
    "patientunitstayid": PATIENT_ID,
    "features": features,
    "agent_results": {},
    "apache_result": None
}

result = run_severity_agent(severity_input)

print("Status:", result["status"])
print("ML:", f'{result["mortality_prediction"]["probability"]:.2%}')
print("Missing features:", result["data_quality"]["n_missing_features"])
print("Top SHAP:", result["shap_explanation"]["top_features"][0]["feature"])
print("APACHE:", result["apache_comparison"]["status"])
print("LLM:", result["llm_analysis"]["status"])

assert result["status"] == "success"
assert abs(result["mortality_prediction"]["probability"] - 0.0254) < 0.001
#assert result["llm_analysis"]["status"] == "not_run"

print("Severity Agent test passed!")
