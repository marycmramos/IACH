"""Unit tests for the orchestration layer without IAedu or real patient data."""
import importlib
from pathlib import Path

import pytest

orch = importlib.import_module("orchestration.orchestrator")


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(orch, "ROOT", tmp_path)
    monkeypatch.setattr(orch, "OUTPUT_DIR", tmp_path / "outputs" / "patients")
    # Avoid requiring the large real eICU CSVs in these unit tests.
    monkeypatch.setattr(orch, "agent_fingerprint", lambda config: "test-fingerprint")
    calls = []

    def runner(name, patient_id):
        calls.append(name)
        return {
            "patient_id": patient_id,
            "status": "success",
            "agent_output": {"patient_id": patient_id, "status": "success", "findings": []},
            "raw_data": {"example": 1},
        }

    monkeypatch.setattr(orch, "execute_agent", runner)
    return calls


def test_three_agents_and_saved_state(isolated):
    state = orch.run_orchestrator(148611)
    assert state["status"] == "success"
    assert state["ready_for_severity"] is True
    assert isolated == ["vital", "laboratory", "clinical"]
    base = orch.OUTPUT_DIR / "148611"
    assert (base / "patient_state.json").is_file()
    assert all((base / f"{name}.json").is_file() for name in isolated)
    assert not (base / "severity.json").exists()


def test_cache_prevents_repeat(isolated):
    orch.run_orchestrator(148611)
    orch.run_orchestrator(148611)
    assert isolated == ["vital", "laboratory", "clinical"]


def test_incomplete_agent_stops_severity(isolated, monkeypatch):
    actual = orch.execute_agent

    def one_fails(name, patient_id):
        if name == "laboratory":
            return {"patient_id": patient_id, "status": "error",
                    "agent_output": {"patient_id": patient_id, "status": "error", "findings": []}}
        return actual(name, patient_id)

    monkeypatch.setattr(orch, "execute_agent", one_fails)
    called = []
    state = orch.run_orchestrator(148611, severity_runner=lambda x: called.append(x),
                                  clean_data_loader=lambda i: {"feature": 1})
    assert state["status"] == "partial"
    assert state["ready_for_severity"] is False
    assert called == []


def test_severity_receives_agent_results_and_features(isolated):
    seen = []

    def severity(payload):
        seen.append(payload)
        return {"status": "success", "patientunitstayid": "148611", "mortality_prediction": {"probability": 0.0254}}

    state = orch.run_orchestrator(148611, severity_runner=severity,
                                  clean_data_loader=lambda i: {"rr_last": 20.0})
    assert state["status"] == "success"
    assert state["severity"]["status"] == "success"
    assert seen[0]["features"] == {"rr_last": 20.0}
    assert set(seen[0]["agent_results"]) == {"vital", "laboratory", "clinical"}
    assert seen[0]["apache_result"] is None
    assert (orch.OUTPUT_DIR / "148611" / "severity.json").is_file()
    assert '"severity"' in (orch.OUTPUT_DIR / "148611" / "patient_state.json").read_text(encoding="utf-8")


def test_bad_patient_id(isolated):
    with pytest.raises(ValueError):
        orch.run_orchestrator(0)
    with pytest.raises(ValueError):
        orch.run_orchestrator(True)


def test_severity_failure_is_reported_in_state(isolated):
    def failing_severity(_):
        raise RuntimeError("LLM unavailable")
    state = orch.run_orchestrator(148611, severity_runner=failing_severity,
                                  clean_data_loader=lambda i: {"rr_last": 20.0})
    assert state["status"] == "partial"
    assert "RuntimeError" in state["severity_error"]
    assert (orch.OUTPUT_DIR / "148611" / "patient_state.json").is_file()
