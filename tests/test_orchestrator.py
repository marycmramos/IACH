"""Unit tests for the current Orchestrator: Severity runs automatically.

All agents, inference, APACHE and IAedu are mocked. No patient CSVs or network.
"""
import importlib
import json
import sys
import types

import pytest

orch = importlib.import_module("orchestration.orchestrator")


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(orch, "ROOT", tmp_path)
    monkeypatch.setattr(orch, "OUTPUT_DIR", tmp_path / "outputs" / "patients")
    monkeypatch.setattr(orch, "agent_fingerprint", lambda config: "test-fingerprint")
    calls = {"agents": [], "severity": [], "apache": [], "features": []}

    def run_agent(name, patient_id):
        calls["agents"].append(name)
        return {
            "patient_id": patient_id,
            "status": "success",
            "agent_output": {"patient_id": patient_id, "status": "success", "findings": []},
            "raw_data": {"example": 1},
        }

    def ml_features(patient_id):
        calls["features"].append(patient_id)
        return {"rr_last": 37.0}

    def apache_loader(patient_id):
        calls["apache"].append(patient_id)
        return {"IV": {"version": "IV", "score": 47, "mortality_probability": 0.008}}

    def severity_agent(payload):
        calls["severity"].append(payload)
        return {
            "agent": "severity", "status": "success",
            "patientunitstayid": str(payload["patientunitstayid"]),
            "mortality_prediction": {"probability": 0.034},
        }

    monkeypatch.setattr(orch, "execute_agent", run_agent)
    monkeypatch.setattr(orch, "load_ml_features", ml_features)
    # The current orchestrator imports these inside run_orchestrator().
    # Mock modules so the unit tests never import real LLM/model code.
    fake_severity = types.ModuleType("agents.severity.severity_agent")
    fake_severity.run_severity_agent = severity_agent
    fake_apache = types.ModuleType("agents.severity.apache_loader")
    fake_apache.load_apache = apache_loader
    monkeypatch.setitem(sys.modules, "agents.severity.severity_agent", fake_severity)
    monkeypatch.setitem(sys.modules, "agents.severity.apache_loader", fake_apache)
    # Compatibility with the earlier root-level import, if still present locally.
    monkeypatch.setitem(sys.modules, "apache_loader", fake_apache)
    return calls


def test_three_agents_and_saved_state(isolated):
    state = orch.run_orchestrator(141765)
    assert state["status"] == "success"
    assert state["ready_for_severity"] is True
    assert isolated["agents"] == ["vital", "laboratory", "clinical"]
    assert len(isolated["severity"]) == 1
    assert isolated["features"] == [141765]
    assert isolated["apache"] == [141765]
    assert isolated["severity"][0]["features"] == {"rr_last": 37.0}
    assert set(isolated["severity"][0]["agent_results"]) == {"vital", "laboratory", "clinical"}
    assert "IV" in isolated["severity"][0]["apache_result"]
    folder = orch.OUTPUT_DIR / "141765"
    assert all((folder / f"{name}.json").is_file() for name in ("vital", "laboratory", "clinical"))
    assert (folder / "severity.json").is_file()
    assert (folder / "patient_state.json").is_file()
    assert json.loads((folder / "severity.json").read_text(encoding="utf-8"))["result"]["status"] == "success"


def test_cache_prevents_repeat(isolated):
    orch.run_orchestrator(141765)
    orch.run_orchestrator(141765)
    assert isolated["agents"] == ["vital", "laboratory", "clinical"]
    assert len(isolated["severity"]) == 2


def test_incomplete_agent_stops_severity(isolated, monkeypatch):
    previous = orch.execute_agent

    def fail_lab(name, patient_id):
        if name == "laboratory":
            return {"patient_id": patient_id, "status": "error",
                    "agent_output": {"patient_id": patient_id, "status": "error", "findings": []}}
        return previous(name, patient_id)

    monkeypatch.setattr(orch, "execute_agent", fail_lab)
    state = orch.run_orchestrator(141765)
    assert state["status"] == "partial"
    assert state["ready_for_severity"] is False
    assert isolated["severity"] == []
    assert isolated["features"] == []
    assert not (orch.OUTPUT_DIR / "141765" / "severity.json").exists()



def test_severity_receives_agent_results_and_features(isolated):
    state = orch.run_orchestrator(141765)
    assert state["status"] == "success"
    assert state["severity"]["status"] == "success"
    assert len(isolated["severity"]) == 1
    payload = isolated["severity"][0]
    assert payload["patientunitstayid"] == 141765
    assert payload["features"] == {"rr_last": 37.0}
    assert set(payload["agent_results"]) == {"vital", "laboratory", "clinical"}
    assert all(data["status"] == "success" for data in payload["agent_results"].values())
    assert payload["apache_result"]["IV"]["score"] == 47
    assert (orch.OUTPUT_DIR / "141765" / "severity.json").is_file()
    assert '"severity"' in (orch.OUTPUT_DIR / "141765" / "patient_state.json").read_text(encoding="utf-8")

def test_no_apache_file_still_runs_severity(isolated, monkeypatch):
    def missing_file(_patient_id):
        raise FileNotFoundError("APACHE CSV not installed")

    monkeypatch.setattr(sys.modules["agents.severity.apache_loader"], "load_apache", missing_file)
    state = orch.run_orchestrator(141765)
    assert state["status"] == "success"
    assert state["severity"]["status"] == "success"
    assert isolated["severity"][0]["apache_result"] is None


def test_no_apache_record_still_runs_severity(isolated, monkeypatch):
    monkeypatch.setattr(sys.modules["agents.severity.apache_loader"], "load_apache", lambda _id: {})
    state = orch.run_orchestrator(141765)
    assert state["status"] == "success"
    assert isolated["severity"][0]["apache_result"] == {}


def test_severity_failure_is_reported_in_state(isolated, monkeypatch):
    def fail(_payload):
        raise RuntimeError("Inference unavailable")

    monkeypatch.setattr(sys.modules["agents.severity.severity_agent"], "run_severity_agent", fail)
    state = orch.run_orchestrator(141765)
    assert state["status"] == "partial"
    assert "RuntimeError" in state["severity_error"]
    assert (orch.OUTPUT_DIR / "141765" / "patient_state.json").is_file()


@pytest.mark.parametrize("bad_id", [0, -1, True, "141765"])
def test_bad_patient_id(isolated, bad_id):
    with pytest.raises(ValueError):
        orch.run_orchestrator(bad_id)


# Additional focused tests for feature extraction, cache invalidation and error handling.

def test_load_ml_features_uses_only_expected_columns(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(orch, "ROOT", tmp_path)
    (tmp_path / "ml_model").mkdir()
    pd.DataFrame([
        {"patientunitstayid": 141765, "rr_last": 37.0, "BUN_max": float("nan"), "target": 1},
        {"patientunitstayid": 156906, "rr_last": 33.0, "BUN_max": 28.0, "target": 0},
    ]).to_csv(tmp_path / "ml_model" / "ml_features_24h.csv", index=False)
    fake_predict = types.ModuleType("ml_model.predict")
    fake_predict.load_model = lambda: (None, ["rr_last", "BUN_max"])
    monkeypatch.setitem(sys.modules, "ml_model.predict", fake_predict)
    features = orch.load_ml_features(141765)
    assert features == {"rr_last": 37.0, "BUN_max": None}
    assert "target" not in features


def test_load_ml_features_rejects_missing_and_duplicate_patients(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(orch, "ROOT", tmp_path)
    (tmp_path / "ml_model").mkdir()
    pd.DataFrame([
        {"patientunitstayid": 141765, "rr_last": 37.0},
        {"patientunitstayid": 141765, "rr_last": 38.0},
    ]).to_csv(tmp_path / "ml_model" / "ml_features_24h.csv", index=False)
    fake_predict = types.ModuleType("ml_model.predict")
    fake_predict.load_model = lambda: (None, ["rr_last"])
    monkeypatch.setitem(sys.modules, "ml_model.predict", fake_predict)
    with pytest.raises(ValueError, match="Expected one ML row"):
        orch.load_ml_features(141765)
    with pytest.raises(ValueError, match="Expected one ML row"):
        orch.load_ml_features(999999)


def test_load_ml_features_rejects_missing_model_columns(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(orch, "ROOT", tmp_path)
    (tmp_path / "ml_model").mkdir()
    pd.DataFrame([{"patientunitstayid": 141765, "rr_last": 37.0}]).to_csv(
        tmp_path / "ml_model" / "ml_features_24h.csv", index=False)
    fake_predict = types.ModuleType("ml_model.predict")
    fake_predict.load_model = lambda: (None, ["rr_last", "missing_column"])
    monkeypatch.setitem(sys.modules, "ml_model.predict", fake_predict)
    with pytest.raises(ValueError, match="missing columns"):
        orch.load_ml_features(141765)


def test_bad_cache_is_not_reused(isolated):
    orch.run_orchestrator(141765)
    p = orch.OUTPUT_DIR / "141765" / "vital.json"
    cached = json.loads(p.read_text(encoding="utf-8"))
    cached["fingerprint"] = "stale-version"
    p.write_text(json.dumps(cached), encoding="utf-8")
    orch.run_orchestrator(141765)
    assert isolated["agents"].count("vital") == 2
    assert isolated["agents"].count("clinical") == 1
    assert isolated["agents"].count("laboratory") == 1


def test_corrupt_json_cache_is_rebuilt(isolated):
    orch.run_orchestrator(141765)
    p = orch.OUTPUT_DIR / "141765" / "vital.json"
    p.write_text("not valid json", encoding="utf-8")
    orch.run_orchestrator(141765)
    assert isolated["agents"].count("vital") == 2


def test_invalid_feature_loader_reported_as_partial(isolated, monkeypatch):
    def no_features(_id):
        return {}
    monkeypatch.setattr(orch, "load_ml_features", no_features)
    state = orch.run_orchestrator(141765)
    assert state["status"] == "partial"
    assert "Missing or invalid ML features" in state["severity_error"]
    assert isolated["severity"] == []
    assert not (orch.OUTPUT_DIR / "141765" / "severity.json").exists()


def test_agent_fingerprint_changes_with_dataset_mtime(tmp_path, monkeypatch):
    import os
    config = {"module": "agents.vital.vital_agent", "sources": ["processed_data/vital.csv"]}
    module_file = tmp_path / "agents" / "vital" / "vital_agent.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("x = 1", encoding="utf-8")
    csv = tmp_path / "processed_data" / "vital.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("id\n1\n", encoding="utf-8")
    monkeypatch.setattr(orch, "ROOT", tmp_path)
    first = orch.agent_fingerprint(config)
    stat = csv.stat()
    os.utime(csv, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    assert first != orch.agent_fingerprint(config)


def test_to_jsonable_rejects_unsupported_type():
    with pytest.raises(TypeError):
        orch.to_jsonable(object())
