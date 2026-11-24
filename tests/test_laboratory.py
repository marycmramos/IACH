"""From repository root: python -m pytest tests/test_explanation.py -v"""
import json
from copy import deepcopy

from agents.explanation.explanation_agent import run_explanation_agent


SEVERITY = {
    "agent": "severity", "status": "success", "patientunitstayid": "141765",
    "prediction_hour": 24,
    "mortality_prediction": {"model": "xgboost_24h", "outcome": "icu_mortality_after_24h", "probability": 0.0254},
    "shap_explanation": {
        "method": "TreeSHAP", "unit": "log-odds", "base_probability": 0.035,
        "top_features": [
            {"feature": "rr_last", "shap_value": 0.3, "was_imputed": False, "original_value": 24},
            {"feature": "age", "shap_value": -0.2, "was_imputed": True, "original_value": None},
        ],
        "all_features": [],
    },
    "data_quality": {"n_features": 53, "n_missing_features": 1, "missing_features": ["age"]},
    "agent_context": {"available_agents": ["vital"], "results": {
        "vital": {"status": "success", "agent_output": {"findings": ["Recorded vital signs were reviewed."]}}
    }},
    "apache_comparison": {"status": "descriptive", "versions": {
        "IV": {"status": "descriptive", "apache_score": 48, "apache_probability": 0.12,
               "ml_probability": 0.0254, "difference_percentage_points": -9.46},
        "IVa": {"status": "descriptive", "apache_score": 46, "apache_probability": 0.10,
                "ml_probability": 0.0254, "difference_percentage_points": -7.46},
    }},
    "llm_analysis": {"status": "success", "assessment": {
        "findings": ["Predicted mortality is approximately 2.54%."],
        "shap_interpretation": [], "apache_assessment": "Descriptive only", "limitations": []}},
    "warnings": ["Research model not clinically validated"],
}


def state():
    return {"patientunitstayid": 141765, "severity_result": deepcopy(SEVERITY)}


REPORT = {
    "mortality_summary": "XGBoost estimates 2.54% ICU mortality.",
    "clinical_summary": "Vital findings were provided.",
    "shap_summary": "rr_last increased and age decreased model log-odds.",
    "overall_explanation": (
        "The estimate is 2.54%; SHAP describes model behavior. "
        "APACHE IV estimates 12.00% and APACHE IVa estimates 10.00% "
        "(descriptive comparisons only)."
    ),
    "limitations": [],
}


class FakeClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return next(self.outputs)


def test_template_uses_real_shap_fields():
    result = run_explanation_agent(state(), use_llm=False)
    assert result["status"] == "success"
    assert result["mortality_prediction"]["probability"] == 0.0254
    assert "rr_last" in result["explanation"]["shap_summary"]
    assert "age" in result["explanation"]["shap_summary"]
    assert "imputed" in result["explanation"]["shap_summary"]


def test_apache_iv_and_iva_present():
    result = run_explanation_agent(state(), use_llm=False)
    assert "APACHE IV estimates" in result["explanation"]["overall_explanation"]
    assert "APACHE IVa estimates" in result["explanation"]["overall_explanation"]
    assert "APACHE comparison is unavailable." not in result["explanation"]["limitations"]


def test_uses_embedded_findings_and_restricts_llm_payload():
    client = FakeClient([REPORT])
    result = run_explanation_agent(state(), llm_client=client)
    assert result["status"] == "success"
    payload = json.loads(client.calls[0][1]["content"])
    assert payload["clinical_agent_findings"]["vital"]["findings"]
    assert "patientunitstayid" not in payload
    assert "severity_llm_assessment" in payload
    assert "shap_explanation" in payload
    assert "shap_assessment" not in payload


def test_absent_apache_is_reported():
    data = state()
    data["severity_result"]["apache_comparison"] = {"status": "unavailable", "versions": {}}
    report = run_explanation_agent(data, use_llm=False)["explanation"]
    assert "APACHE comparison is unavailable." in report["limitations"]


def test_missing_shap_not_invented():
    data = state()