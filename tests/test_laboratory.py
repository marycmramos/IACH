"""Laboratory Agent tests. No real IAedu requests."""
import json
import pytest

from agents.laboratory import laboratory_agent as lab


def _sample():
    # Choose an ICU stay with at least one numerical lab evaluation.
    for patient_id in lab.lab_df.index:
        patient_id = int(patient_id)
        evaluations = lab.evaluate_lab_ranges(lab.get_lab_data(patient_id))
        if evaluations:
            return patient_id, evaluations
    pytest.skip("Sem pacientes com observações laboratoriais")


def test_lab_missing_patient():
    result = lab.run_lab_agent(999999999)
    assert result.status == "error"
    assert result.agent_output.status == "error"


def test_lab_valid_selection(monkeypatch):
    patient_id, evaluations = _sample()
    monkeypatch.setattr(lab, "ask_iaedu", lambda prompt: json.dumps({
        "patient_id": patient_id, "status": "success", "findings": evaluations[:2]
    }))
    result = lab.run_lab_agent(patient_id)
    assert result.status == "success"
    assert [f.model_dump() for f in result.agent_output.findings] == evaluations[:2]
    assert all(x.flag == "unknown" for x in result.agent_output.findings) if not lab.LAB_RANGES_VALIDATED else True


def test_lab_rejects_fabricated_value(monkeypatch):
    patient_id, evaluations = _sample()
    fake = {**evaluations[0], "value": evaluations[0]["value"] + 9999.0}
    monkeypatch.setattr(lab, "ask_iaedu", lambda prompt: json.dumps({
        "patient_id": patient_id, "status": "success", "findings": [fake]
    }))
    result = lab.run_lab_agent(patient_id)
    assert result.status == "error"
    assert result.agent_output.findings == []
