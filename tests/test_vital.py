"""Vital Agent tests. No real IAedu requests."""
import json
import pytest

from agents.vital import vital_agent as vital


def _sample():
    for patient_id in vital.vital_df.index:
        patient_id = int(patient_id)
        evaluations = vital.evaluate_vital_ranges(vital.get_vital_data(patient_id))
        if evaluations:
            return patient_id, evaluations
    pytest.skip("Sem pacientes com observações de sinais vitais")


def test_vital_missing_patient():
    result = vital.run_vital_agent(999999999)
    assert result.status == "error"
    assert result.agent_output.status == "error"


def test_vital_valid_selection(monkeypatch):
    patient_id, evaluations = _sample()
    monkeypatch.setattr(vital, "ask_iaedu", lambda prompt: json.dumps({
        "patient_id": patient_id, "status": "success", "findings": evaluations[:2]
    }))
    result = vital.run_vital_agent(patient_id)
    assert result.status == "success"
    assert [f.model_dump() for f in result.agent_output.findings] == evaluations[:2]
    assert all(x.flag == "unknown" for x in result.agent_output.findings) if not vital.RANGES_VALIDATED else True


def test_vital_rejects_fabricated_value(monkeypatch):
    patient_id, evaluations = _sample()
    fake = {**evaluations[0], "value": evaluations[0]["value"] + 9999.0}
    monkeypatch.setattr(vital, "ask_iaedu", lambda prompt: json.dumps({
        "patient_id": patient_id, "status": "success", "findings": [fake]
    }))
    result = vital.run_vital_agent(patient_id)
    assert result.status == "error"
    assert result.agent_output.findings == []
