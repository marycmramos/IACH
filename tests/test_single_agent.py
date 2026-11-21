"""Offline tests for IACH Single-Agent Baseline.

Run from the project root:
    python -m pytest -q tests/test_single_agent.py

No IAedu/API calls are performed.
"""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from agents.single_agent import single_agent as sa

PATIENT_ID = 176730
PROBABILITY = 0.011178342625498772


def severity_fixture():
    return {
        "status": "success",
        "patientunitstayid": str(PATIENT_ID),
        "mortality_prediction": {"model": "xgboost_24h", "probability": PROBABILITY},
        "shap_explanation": {
            "top_features": [{
                "feature": "age", "shap_value": -0.1,
                "original_value": 72, "processed_value": 72,
                "was_imputed": False, "direction": "decreases",
            }]
        },
        "apache_comparison": {"status": "unavailable"},
        "data_quality": {"n_features": 53, "n_missing_features": 0},
    }


def evidence_fixture():
    return {
        "clinical_data": {
            "vital_signs": {"hr_mean": 117.36},
            "laboratory": {"BUN_last": 12.0},
            "clinical": {"age": 72},
        },
        "mortality_prediction": {"model": "xgboost_24h", "probability": PROBABILITY},
        "shap_explanation": copy.deepcopy(severity_fixture()["shap_explanation"]),
        "apache_comparison": {"status": "unavailable"},
        "data_quality": {"n_features": 53, "n_missing_features": 0},
    }


def report_fixture():
    return {
        "mortality_summary": "1.12%",
        "clinical_summary": "Observed age was 72 years.",
        "shap_summary": "SHAP values are in log-odds; age contributed -0.1.",
        "overall_explanation": "XGBoost estimates 1.12%. APACHE comparison is unavailable.",
        "limitations": ["The model is not clinically validated."],
    }


@pytest.fixture
def mocked_inputs(monkeypatch):
    """No patient file or IAedu access during unit tests."""
    monkeypatch.setattr(sa, "_payload", lambda patient_id: (
        copy.deepcopy(evidence_fixture()), copy.deepcopy(severity_fixture())
    ))


def test_valid_llm_response_succeeds_first_time(mocked_inputs):
    calls = []

    def llm(messages):
        calls.append(messages)
        return json.dumps(report_fixture())

    result = sa.run_single_agent(PATIENT_ID, llm_client=llm)

    assert result["status"] == "success"
    assert result["llm_attempts"] == 1
    assert result["explanation"]["mortality_summary"] == "1.12%"
    assert len(calls) == 1
    payload = json.loads(calls[0][1]["content"])
    assert set(payload["clinical_data"]) == {"vital_signs", "laboratory", "clinical"}
    assert "agent_results" not in payload
    assert "explanation" not in payload


def test_wrong_probability_is_retried(mocked_inputs):
    calls = []

    def llm(messages):
        calls.append(1)
        report = report_fixture()
        if len(calls) == 1:
            report["mortality_summary"] = "30.00%"
        return report

    result = sa.run_single_agent(PATIENT_ID, llm_client=llm)

    assert result["status"] == "success"
    assert result["llm_attempts"] == 2
    assert len(calls) == 2


def test_invalid_json_stops_after_three_attempts(mocked_inputs):
    calls = []

    def llm(messages):
        calls.append(1)
        return "not valid json {"

    result = sa.run_single_agent(PATIENT_ID, llm_client=llm, max_attempts=3)

    assert result["status"] == "error"
    assert result["error"]["type"] == "InvalidLLMOutput"
    assert len(calls) == 3


def test_429_does_not_retry(mocked_inputs):
    calls = []

    def llm(messages):
        calls.append(1)
        raise RuntimeError("IAedu error: Rate limit reached (429)")

    result = sa.run_single_agent(PATIENT_ID, llm_client=llm)

    assert result["status"] == "error"
    assert result["error"]["type"] == "LLMRequestError"
    assert len(calls) == 1


def test_shap_percentage_is_rejected(mocked_inputs):
    def llm(messages):
        report = report_fixture()
        report["shap_summary"] = "SHAP contributes 12% to mortality."
        return report

    result = sa.run_single_agent(PATIENT_ID, llm_client=llm, max_attempts=1)
    assert result["status"] == "error"
    assert result["error"]["type"] == "InvalidLLMOutput"


def test_missing_severity_returns_error_before_calling_llm(monkeypatch):
    def missing(patient_id):
        raise FileNotFoundError("severity.json is missing")

    monkeypatch.setattr(sa, "_payload", missing)

    def forbidden_llm(messages):
        pytest.fail("LLM must not be called if inputs are missing")

    result = sa.run_single_agent(PATIENT_ID, llm_client=forbidden_llm)
    assert result["status"] == "error"
    assert result["error"]["type"] == "FileNotFoundError"


@pytest.mark.parametrize("patient_id", [0, -4, True, "176730"])
def test_invalid_patient_ids_rejected(patient_id):
    with pytest.raises(ValueError, match="patient_id"):
        sa.run_single_agent(patient_id, llm_client=lambda messages: report_fixture())


def test_imputed_shap_is_disclosed(mocked_inputs, monkeypatch):
    evidence, severity = evidence_fixture(), severity_fixture()
    evidence["shap_explanation"]["top_features"][0]["was_imputed"] = True
    severity["shap_explanation"]["top_features"][0]["was_imputed"] = True
    monkeypatch.setattr(sa, "_payload", lambda patient_id: (evidence, severity))
    result = sa.run_single_agent(PATIENT_ID, llm_client=lambda messages: report_fixture())
    assert result["status"] == "success"
    assert any("imputed" in note.lower() for note in result["explanation"]["limitations"])


def test_csv_row_read_and_missing_values(tmp_path):
    csv_path = tmp_path / "clinical.csv"
    pd.DataFrame([
        {"patientunitstayid": PATIENT_ID, "age": 72, "note": "", "creatinine": None},
        {"patientunitstayid": 999, "age": 54, "note": "test", "creatinine": 1.1},
    ]).to_csv(csv_path, index=False)

    row = sa._read_episode_row(csv_path, PATIENT_ID)
    assert row["age"] == 72
    assert row["creatinine"] is None
    assert "patientunitstayid" not in row

    with pytest.raises(ValueError, match="found 0"):
        sa._read_episode_row(csv_path, 123456)


def test_severity_patient_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "ROOT", tmp_path)
    folder = tmp_path / "outputs" / "patients" / str(PATIENT_ID)
    folder.mkdir(parents=True)
    severity = severity_fixture()
    severity["patientunitstayid"] = "999"
    (folder / "severity.json").write_text(json.dumps({"result": severity}), encoding="utf-8")

    with pytest.raises(ValueError, match="mismatched"):
        sa._load_severity(PATIENT_ID)
