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
    data["severity_result"]["shap_explanation"]["top_features"] = []
    result = run_explanation_agent(data, use_llm=False)
    assert "not provided" in result["explanation"]["shap_summary"]


def test_absent_severity_error():
    assert run_explanation_agent({"patientunitstayid": 1}, use_llm=False)["status"] == "error"


def test_invalid_probability_error():
    data = state()
    data["severity_result"]["mortality_prediction"]["probability"] = 1.2
    assert run_explanation_agent(data, use_llm=False)["status"] == "error"


def test_episode_mismatch_error():
    data = state()
    data["patientunitstayid"] = 999
    assert run_explanation_agent(data, use_llm=False)["status"] == "error"


def test_integration_agent_results_severity():
    data = {"agent_results": {"severity": deepcopy(SEVERITY)}}
    result = run_explanation_agent(data, use_llm=False)
    assert result["status"] == "success"
    assert result["patientunitstayid"] == "141765"


def test_llm_receives_severity_assessment_and_apache():
    client = FakeClient([REPORT])
    result = run_explanation_agent(state(), llm_client=client)
    payload = json.loads(client.calls[0][1]["content"])
    assert payload["severity_llm_assessment"]["apache_assessment"] == "Descriptive only"
    assert payload["apache_comparison"]["status"] == "descriptive"
    assert result["generation_method"] == "llm"


def test_invalid_json_retries_three_times():
    client = FakeClient(["not json", "not json", "not json"])
    result = run_explanation_agent(state(), llm_client=client)
    assert result["status"] == "error"
    assert len(client.calls) == 3


def test_invalid_structure_retries_then_succeeds():
    client = FakeClient([{"mortality_summary": "only"}, REPORT])
    result = run_explanation_agent(state(), llm_client=client)
    assert result["status"] == "success"
    assert len(client.calls) == 2


def test_network_failure_does_not_retry():
    def broken(_messages):
        raise requests.exceptions.ConnectionError("Sensitive server message")
    import requests
    result = run_explanation_agent(state(), llm_client=broken)
    assert result["status"] == "error"
    assert "Sensitive server message" not in result["error"]["message"]


def test_no_llm_api_needed_for_template():
    assert run_explanation_agent(state(), use_llm=False)["generation_method"] == "template"


def test_rejects_invalid_shap_numeric_value():
    data = state()
    data['severity_result']['shap_explanation']['top_features'][0]['shap_value'] = float('nan')
    result = run_explanation_agent(data, use_llm=False)
    assert result['status'] == 'error'


def test_rejects_invalid_shap_unit():
    data = state()
    data['severity_result']['shap_explanation']['unit'] = 'probability'
    result = run_explanation_agent(data, use_llm=False)
    assert result['status'] == 'error'


def test_rejects_invalid_shap_imputation_flag():
    data = state()
    data['severity_result']['shap_explanation']['top_features'][0]['was_imputed'] = 'false'
    result = run_explanation_agent(data, use_llm=False)
    assert result['status'] == 'error'


def test_rejects_invalid_apache_probability():
    data = state()
    data['severity_result']['apache_comparison']['versions']['IV']['apache_probability'] = 1.5
    result = run_explanation_agent(data, use_llm=False)
    assert result['status'] == 'error'


def test_rejects_apache_inconsistent_status():
    data = state()
    data['severity_result']['apache_comparison']['status'] = 'unavailable'
    result = run_explanation_agent(data, use_llm=False)
    assert result['status'] == 'error'


def test_rejects_wrong_mortality_percentage_then_accepts_correct_one():
    incorrect = {**REPORT, 'mortality_summary': 'The XGBoost estimate is 25.4%.'}
    client = FakeClient([incorrect, REPORT])
    result = run_explanation_agent(state(), llm_client=client)
    assert result['status'] == 'success'
    assert len(client.calls) == 2


def test_rejects_shap_percentages():
    bad = {**REPORT, 'shap_summary': 'rr_last increased mortality by 30%.'}
    client = FakeClient([bad, bad, bad])
    result = run_explanation_agent(state(), llm_client=client)
    assert result['status'] == 'error'
    assert len(client.calls) == 3


def test_payload_omits_full_shap_vector_and_patient_id():
    data = state()
    data['severity_result']['shap_explanation']['all_features'] = [{'sensitive': 'do not send'}]
    client = FakeClient([REPORT])
    result = run_explanation_agent(data, llm_client=client)
    assert result['status'] == 'success'
    payload = json.loads(client.calls[0][1]['content'])
    assert 'all_features' not in payload['shap_explanation']
    assert 'patientunitstayid' not in payload


def test_partial_explicit_findings_falls_back_to_embedded():
    data = state()
    data['agent_findings'] = {'clinical': {'status': 'success', 'agent_output': {'findings': ['Clinical record reviewed.']}}}
    client = FakeClient([REPORT])
    result = run_explanation_agent(data, llm_client=client)
    assert result['status'] == 'success'
    payload = json.loads(client.calls[0][1]['content'])
    assert 'vital' in payload['clinical_agent_findings']
    assert 'clinical' in payload['clinical_agent_findings']


def test_apache_single_version_allowed():
    data = state()
    del data['severity_result']['apache_comparison']['versions']['IVa']
    result = run_explanation_agent(data, use_llm=False)
    assert result['status'] == 'success'
    assert 'APACHE IV estimates' in result['explanation']['overall_explanation']
    assert 'APACHE IVa estimates' not in result['explanation']['overall_explanation']


def test_rejects_apache_ml_probability_mismatch():
    data = state()
    data['severity_result']['apache_comparison']['versions']['IV']['ml_probability'] = 0.45
    result = run_explanation_agent(data, use_llm=False)
    assert result['status'] == 'error'


def test_critic_revision_context_in_llm_payload():
    previous = deepcopy(REPORT)
    client = FakeClient([REPORT])
    result = run_explanation_agent(state(), llm_client=client, revision_context={
        "previous_explanation": previous,
        "issues": [{"code": "RISK_MISMATCH", "reason": "Correct probability", "suggestion": "Use 2.54%"}],
    })
    assert result["status"] == "success"
    payload = json.loads(client.calls[0][1]["content"])
    assert payload["revision_request"]["previous_explanation"] == previous
    assert payload["revision_request"]["critic_issues"][0]["code"] == "RISK_MISMATCH"
    assert "REVISION" in client.calls[0][0]["content"]
    assert "patientunitstayid" not in payload


def test_invalid_critic_feedback_rejected():
    bad = run_explanation_agent(state(), use_llm=False, revision_context={"previous_explanation": REPORT, "issues": []})
    assert bad["status"] == "error"
    bad2 = run_explanation_agent(state(), use_llm=True, llm_client=FakeClient([REPORT]), revision_context={"previous_explanation": REPORT, "issues": []})
    assert bad2["status"] == "error"


def test_revision_keeps_probability_guard():
    bad = {**REPORT, "mortality_summary": "The model risk is 25.4%."}
    client = FakeClient([bad, REPORT])
    outcome = run_explanation_agent(state(), llm_client=client, revision_context={
        "previous_explanation": deepcopy(REPORT),
        "issues": [{"code": "MISMATCH", "reason": "Correct probability"}],
    })
    assert outcome["status"] == "success"
    assert len(client.calls) == 2


def test_rejects_missing_apache_percentage_in_llm_report():
    missing = deepcopy(REPORT)
    missing["overall_explanation"] = (
        "The estimate is 2.54%; SHAP describes model behavior. "
        "APACHE IVa estimates 10.00% (descriptive comparison only)."
    )
    client = FakeClient([missing, missing, missing])
    result = run_explanation_agent(state(), llm_client=client)
    assert result["status"] == "error"
    assert len(client.calls) == 3


def test_rejects_apache_fraction_or_wrong_percentage():
    wrong = deepcopy(REPORT)
    wrong["overall_explanation"] = (
        "The estimate is 2.54%; SHAP describes model behavior. "
        "APACHE IV estimates 0.12% and APACHE IVa estimates 10.00% "
        "(descriptive comparisons only)."
    )
    client = FakeClient([wrong, wrong, wrong])
    result = run_explanation_agent(state(), llm_client=client)
    assert result["status"] == "error"
    assert len(client.calls) == 3
