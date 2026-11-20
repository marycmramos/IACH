
import json

from agents.explanation.explanation_agent import run_explanation_agent

patient_state = {
    "patientunitstayid": "12345",
    "agent_results": {
        "severity": {
            "agent": "severity",
            "status": "success",
            "patientunitstayid": "12345",
            "mortality_prediction": {
                "model": "xgboost_24h",
                "probability": 0.0254
            },
            "shap_explanation": {
                "method": "TreeSHAP",
                "unit": "log-odds",
                "base_probability": 0.03,
                "top_features": [
                    {
                        "feature": "rr_last",
                        "shap_value": 0.3,
                        "was_imputed": False,
                        "original_value": 24
                    },
                    {
                        "feature": "age",
                        "shap_value": -0.2,
                        "was_imputed": False,
                        "original_value": 65
                    }
                ]
            },
            "apache_comparison": {
                "status": "unavailable",
                "versions": {}
            },
            "llm_analysis": {
                "status": "not_run",
                "assessment": None
            }
        }
    }
}

result = run_explanation_agent(
    patient_state,
    use_llm=True
)

print(json.dumps(result, indent=2, ensure_ascii=False))
