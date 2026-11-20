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
    def explanation_agent(payload, use_llm=True):
        calls.setdefault("explanation", []).append(payload)
        return {
            "agent": "explanation", "status": "success",
            "patientunitstayid": payload["patientunitstayid"],
            "explanation": {
                "mortality_summary": "The model mortality risk is 3.40%.",
                "clinical_summary": "Clinical observations are available.",
                "shap_summary": "SHAP summarizes the model behavior.",
                "overall_explanation": "Research report only.",
                "limitations": ["Not validated for clinical use."],
            }
        }

    def critic_agent(payload):
        calls.setdefault("critic", []).append(payload)
        return {
            "agent": "critic", "status": "approved", "approved": True,
            "patientunitstayid": str(payload["patientunitstayid"]), "issues": [],
            "deterministic_check": "completed", "llm_check": "success",
            "review_summary": "No issues found.", "error_type": None,
        }

    fake_explanation = types.ModuleType("agents.explanation.explanation_agent")
    fake_explanation.run_explanation_agent = explanation_agent
    fake_critic = types.ModuleType("agents.critic.critic_agent")
    fake_critic.run_critic_agent = critic_agent
    monkeypatch.setitem(sys.modules, "agents.explanation.explanation_agent", fake_explanation)
    monkeypatch.setitem(sys.modules, "agents.critic.critic_agent", fake_critic)
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


def test_explanation_and_critic_integration(isolated):
    state = orch.run_orchestrator(141765)
    assert state["status"] == "success"
    assert len(isolated["explanation"]) == 1
    assert len(isolated["critic"]) == 1
    assert "mortality_summary:" in isolated["critic"][0]["explanation_text"]
    assert "overall_explanation:" in isolated["critic"][0]["explanation_text"]
    assert "limitations:" in isolated["critic"][0]["explanation_text"]
    assert (orch.OUTPUT_DIR / "141765" / "explanation.json").is_file()
    assert (orch.OUTPUT_DIR / "141765" / "critic.json").is_file()


def test_needs_revision_keeps_explanation_and_marks_partial(isolated, monkeypatch):
    def needs_revision(payload):
        return {"agent": "critic", "status": "needs_revision", "approved": False,
                "patientunitstayid": str(payload["patientunitstayid"]), "issues": []}
    monkeypatch.setattr(sys.modules["agents.critic.critic_agent"], "run_critic_agent", needs_revision)
    state = orch.run_orchestrator(141765)
    assert state["status"] == "partial"
    assert state["critic"]["status"] == "needs_revision"
    assert state["explanation"]["status"] == "success"


def test_critic_exception_does_not_erase_explanation(isolated, monkeypatch):
    def fail(_payload):
        raise RuntimeError("Critic unavailable")
    monkeypatch.setattr(sys.modules["agents.critic.critic_agent"], "run_critic_agent", fail)
    state = orch.run_orchestrator(141765)
    assert state["status"] == "partial"
    assert "critic_error" in state
    assert state["explanation"]["status"] == "success"


def test_critic_adapter_rejects_missing_report_sections():
    with pytest.raises(ValueError, match="Missing or empty"):
        orch.explanation_to_critic_text({
            "status": "success", "patientunitstayid": 42,
            "explanation": {"mortality_summary": "OK", "clinical_summary": "OK",
                            "shap_summary": "OK", "overall_explanation": " " ,"limitations": []},
        }, 42)


def test_critic_adapter_rejects_different_patient():
    with pytest.raises(ValueError, match="different patient ID"):
        orch.explanation_to_critic_text({"status": "success", "patientunitstayid": 2}, 1)


def test_explanation_mutation_cannot_change_severity_used_by_critic(isolated, monkeypatch):
    def mutating_explanation(payload, use_llm=True):
        payload["severity_result"]["shap_explanation"] = {"unit": "log-odds"}
        return {"status": "success", "patientunitstayid": payload["patientunitstayid"],
                "explanation": {"mortality_summary": "The model mortality risk is 3.40%.",
                                "clinical_summary": "Clinical observations.",
                                "shap_summary": "SHAP explains model behavior.",
                                "overall_explanation": "Research summary.", "limitations": []}}
    monkeypatch.setattr(sys.modules["agents.explanation.explanation_agent"],
                        "run_explanation_agent", mutating_explanation)
    state = orch.run_orchestrator(141765)
    assert state["status"] == "success"
    assert "shap_explanation" not in state["severity"]
    assert "shap_explanation" not in isolated["critic"][0]["severity_result"]

# Critic-driven review loop: all model calls are mocked.

def _install_review_fakes(isolated, monkeypatch, critic_statuses, revision_failure=False, unchanged=False):
    explanation_mod = sys.modules["agents.explanation.explanation_agent"]
    critic_mod = sys.modules["agents.critic.critic_agent"]
    report_calls = []
    critic_calls = []

    def explanation(payload, use_llm=True, revision_context=None):
        report_calls.append(revision_context)
        if revision_context is not None and revision_failure:
            return {"agent": "explanation", "status": "error", "patientunitstayid": payload["patientunitstayid"]}
        if revision_context and unchanged:
            report = revision_context["previous_explanation"].copy()
        else:
            n = len(report_calls)
            report = {
                "mortality_summary": "The model mortality risk is 3.40%.",
                "clinical_summary": f"Clinical observations, version {n}.",
                "shap_summary": "SHAP values describe model behavior.",
                "overall_explanation": "Academic prototype, not validated clinically.",
                "limitations": ["Review by investigator required."],
            }
        return {"agent": "explanation", "status": "success", "patientunitstayid": payload["patientunitstayid"], "explanation": report}

    def critic(payload):
        critic_calls.append(payload)
        status = critic_statuses[min(len(critic_calls)-1, len(critic_statuses)-1)]
        return {
            "agent": "critic", "status": status, "approved": status == "approved",
            "patientunitstayid": str(payload["patientunitstayid"]),
            "issues": [] if status != "needs_revision" else [{
                "code": "RISK_MISMATCH", "claim": "Claim is unsupported", "reason": "Check evidence",
                "suggestion": "Correct the explanation", "evidence": "mortality_prediction",
            }],
            "deterministic_check": "completed", "llm_check": "success",
            "review_summary": "Checked", "error_type": None,
        }

    monkeypatch.setattr(explanation_mod, "run_explanation_agent", explanation)
    monkeypatch.setattr(critic_mod, "run_critic_agent", critic)
    return report_calls, critic_calls


def test_revision_approved_after_feedback(isolated, monkeypatch):
    explanation_calls, critic_calls = _install_review_fakes(isolated, monkeypatch, ["needs_revision", "approved"])
    state = orch.run_orchestrator(141765)
    assert state["status"] == "success"
    assert len(explanation_calls) == len(critic_calls) == 2
    assert explanation_calls[0] is None
    assert explanation_calls[1]["issues"][0]["code"] == "RISK_MISMATCH"
    assert "version 1" in explanation_calls[1]["previous_explanation"]["clinical_summary"]
    assert state["review_cycle"]["stop_reason"] == "approved"
    assert state["review_cycle"]["revisions_used"] == 1
    location = orch.ROOT / state["review_cycle"]["history_path"]
    assert (location / "round_00.json").exists() and (location / "round_01.json").exists()
    latest = json.loads((orch.OUTPUT_DIR / "141765" / "explanation.json").read_text())["result"]
    assert "version 2" in latest["explanation"]["clinical_summary"]


def test_revision_cap_preserves_last_reviewed(isolated, monkeypatch):
    explanations, critics = _install_review_fakes(isolated, monkeypatch, ["needs_revision"])
    state = orch.run_orchestrator(141765, max_revisions=2)
    assert state["status"] == "partial"
    assert state["review_cycle"]["stop_reason"] == "revision_limit_reached"
    assert len(explanations) == len(critics) == 3
    assert len(state["review_cycle"]["history"]) == 3
    assert state["critic"]["status"] == "needs_revision"


def test_review_error_does_not_regenerate(isolated, monkeypatch):
    explanations, critics = _install_review_fakes(isolated, monkeypatch, ["review_error"])
    state = orch.run_orchestrator(141765)
    assert state["status"] == "partial"
    assert len(explanations) == len(critics) == 1
    assert state["review_cycle"]["stop_reason"] == "review_error"


def test_revision_failure_preserves_initial_review(isolated, monkeypatch):
    explanations, critics = _install_review_fakes(isolated, monkeypatch, ["needs_revision"], revision_failure=True)
    state = orch.run_orchestrator(141765)
    assert state["status"] == "partial"
    assert state["review_cycle"]["stop_reason"] == "explanation_revision_error"
    assert len(explanations) == 2 and len(critics) == 1
    assert "version 1" in state["explanation"]["explanation"]["clinical_summary"]
    assert state["critic"]["status"] == "needs_revision"


def test_identical_revision_stops(isolated, monkeypatch):
    explanations, critics = _install_review_fakes(isolated, monkeypatch, ["needs_revision"], unchanged=True)
    state = orch.run_orchestrator(141765)
    assert state["status"] == "partial"
    assert state["review_cycle"]["stop_reason"] == "unchanged_revision"
    assert len(explanations) == 2 and len(critics) == 1
    loc = orch.ROOT / state["review_cycle"]["history_path"]
    assert json.loads((loc / "round_01.json").read_text())["critic"] is None


def test_zero_revisions_and_invalid_budget(isolated, monkeypatch):
    explanations, critics = _install_review_fakes(isolated, monkeypatch, ["needs_revision"])
    state = orch.run_orchestrator(141765, max_revisions=0)
    assert len(explanations) == len(critics) == 1
    assert state["review_cycle"]["stop_reason"] == "revision_limit_reached"
    for bad in [-1, 11, True, "2"]:
        with pytest.raises(ValueError, match="max_revisions"):
            orch.run_orchestrator(141765, max_revisions=bad)


def test_previous_runs_retained(isolated, monkeypatch):
    _install_review_fakes(isolated, monkeypatch, ["approved"])
    a = orch.run_orchestrator(141765)["review_cycle"]["history_path"]
    b = orch.run_orchestrator(141765)["review_cycle"]["history_path"]
    assert a != b
    assert (orch.ROOT / a / "round_00.json").exists()
    assert (orch.ROOT / b / "round_00.json").exists()


def test_old_top_level_approval_not_reused_after_critic_failure(isolated, monkeypatch):
    _install_review_fakes(isolated, monkeypatch, ["approved"])
    orch.run_orchestrator(141765)
    critic_file = orch.OUTPUT_DIR / "141765" / "critic.json"
    assert critic_file.exists()
    def broken(_payload):
        raise RuntimeError("Critic offline")
    monkeypatch.setattr(sys.modules["agents.critic.critic_agent"], "run_critic_agent", broken)
    state = orch.run_orchestrator(141765)
    assert state["status"] == "partial"
    assert not critic_file.exists()
