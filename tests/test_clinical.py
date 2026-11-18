"""Clinical Agent tests. No real IAedu requests."""
import json

from agents.clinical import clinical_agent as clinical


def test_clinical_missing_patient():
    result = clinical.run_clinical_agent(999999999)
    assert result.status == "error"
    assert result.agent_output.status == "error"


def test_clinical_valid_selection(monkeypatch):
    patient_id = int(clinical.clinical_df.index[0])
    evidence = clinical.build_clinical_evidence(clinical.get_clinical_data(patient_id))
    if not evidence:
        import pytest
        pytest.skip("Primeiro paciente sem evidência clínica")

    selected = evidence[:2]
    monkeypatch.setattr(clinical, "ask_iaedu", lambda prompt: json.dumps({
        "patient_id": patient_id, "status": "success", "findings": selected
    }))
    result = clinical.run_clinical_agent(patient_id)
    assert result.status == "success"
    assert [f.model_dump() for f in result.agent_output.findings] == selected


def test_clinical_rejects_invented_finding(monkeypatch):
    patient_id = int(clinical.clinical_df.index[0])
    monkeypatch.setattr(clinical, "ask_iaedu", lambda prompt: json.dumps({
        "patient_id": patient_id, "status": "success",
        "findings": [{"category": "admission_diagnosis", "value": "INVENTED DIAGNOSIS"}]
    }))
    result = clinical.run_clinical_agent(patient_id)
    if not result.evidence:
        import pytest
        pytest.skip("Paciente sem evidência; não chama IAedu")
    assert result.status == "error"
    assert result.agent_output.findings == []
