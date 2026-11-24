"""Offline human-review tests; no API credentials or patient CSVs required."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from orchestration import orchestrator as orch
from orchestration import human_review as human


@pytest.fixture
def review_env(monkeypatch, tmp_path):
    monkeypatch.setattr(orch, "OUTPUT_DIR", tmp_path / "outputs" / "patients")
    severity = {
        "agent": "severity", "status": "success", "patientunitstayid": "156906",
        "mortality_prediction": {"model": "xgboost_24h", "probability": 0.02536257542669773},
        "shap_explanation": {"unit": "log_odds", "top_features": []},
        "apache_comparison": {"status": "unavailable", "versions": {}},
        "llm_analysis": None,
    }
    explanation = {
        "agent": "explanation", "status": "success", "patientunitstayid": 156906,
        "explanation": {
            "mortality_summary": "The model estimates mortality of 2.54%.",
            "clinical_summary": "Synthetic clinical observations supplied.",
            "shap_summary": "SHAP details not provided.",
            "overall_explanation": "Synthetic research summary.",
            "limitations": ["Not clinically validated."],
        },
    }
    critic = {"agent": "critic", "patientunitstayid": "156906", "status": "approved", "approved": True, "issues": []}
    state = {
        "patientunitstayid": 156906, "severity": severity,
        "explanation": explanation, "critic": critic,
        "agents": {}, "review_cycle": {"run_id": "run_alpha"},
    }
    path = orch.OUTPUT_DIR / "156906" / "patient_state.json"
    orch.save_json(path, state)
    return state


def test_session_creation_and_resume(review_env):
    a = human.get_review(156906)
    b = human.get_review(156906)
    assert a == b
    assert a["status"] == "pending"
    assert len(a["versions"]) == 1
    assert (orch.OUTPUT_DIR / "156906" / "human_reviews" / "run_alpha" / "version_00.json").exists()


def test_approval_is_human_explicit_and_persistent(review_env):
    session = human.approve_review(156906, "investigator-A")
    assert session["status"] == "approved"
    assert session["approved_version"] == 0
    assert session["decisions"][-1]["reviewer"] == "investigator-A"
    assert human.get_review(156906)["status"] == "approved"
    with pytest.raises(human.HumanReviewError, match="already approved"):
        human.approve_review(156906)


def test_cannot_approve_needs_revision(review_env):
    path = orch.OUTPUT_DIR / "156906" / "patient_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["critic"]["status"] = "needs_revision"
    state["critic"]["approved"] = False
    orch.save_json(path, state)
    with pytest.raises(human.HumanReviewError, match="not approved"):
        human.approve_review(156906)
    assert human.get_review(156906)["status"] == "pending"


def test_revision_uses_existing_evidence_and_preserves_previous_version(review_env):
    seen = {}

    def fake_explanation(payload, **kwargs):
        seen["payload"] = payload
        seen["revision_context"] = kwargs["revision_context"]
        result = deepcopy(review_env["explanation"])
        result["explanation"]["clinical_summary"] += " Additional detail requested by reviewer."
        return result

    def fake_critic(payload):
        seen["critic_input"] = payload
        return {"agent": "critic", "patientunitstayid": "156906", "status": "approved", "approved": True, "issues": []}

    session = human.request_revision(156906, "Please clarify the clinical summary", "investigator-A", fake_explanation, fake_critic)
    assert session["current_version"] == 1
    assert len(session["versions"]) == 2
    assert session["status"] == "pending"
    assert seen["revision_context"]["issues"][0]["code"] == "HUMAN_FEEDBACK"
    assert seen["payload"]["severity_result"]["mortality_prediction"]["probability"] == 0.02536257542669773
    assert "Additional detail" in seen["critic_input"]["explanation_text"]
    assert session["versions"][0]["explanation"]["explanation"]["clinical_summary"] != session["versions"][1]["explanation"]["explanation"]["clinical_summary"]
    assert (orch.OUTPUT_DIR / "156906" / "human_reviews" / "run_alpha" / "version_01.json").exists()
    assert human.approve_review(156906)["approved_version"] == 1


def test_multiple_revisions_no_auto_critic_limit(review_env):
    for index in range(3):
        def explain(_payload, **kwargs):
            result = deepcopy(review_env["explanation"])
            result["explanation"]["clinical_summary"] += f" Version {index}."
            return result
        def critic(_payload):
            return {"patientunitstayid": "156906", "status": "needs_revision", "approved": False, "issues": [{"code": "test"}]}
        session = human.request_revision(156906, f"Change {index}", explanation_runner=explain, critic_runner=critic)
        assert session["current_version"] == index + 1
    assert len(session["versions"]) == 4
    with pytest.raises(human.HumanReviewError, match="not approved"):
        human.approve_review(156906)


def test_failed_revision_preserves_current_version_and_audits_failure(review_env):
    def fail(_payload, **kwargs):
        raise RuntimeError("Secret HTTP token")
    with pytest.raises(human.HumanReviewError, match="RuntimeError") as error:
        human.request_revision(156906, "Make simpler", explanation_runner=fail, critic_runner=lambda p: None)
    assert "Secret HTTP token" not in str(error.value)
    session = human.get_review(156906)
    assert session["current_version"] == 0
    assert len(session["versions"]) == 1
    assert session["decisions"][0]["result"] == "failed"


def test_invalid_feedback_and_unchanged_report(review_env):
    with pytest.raises(human.HumanReviewError, match="Feedback"):
        human.request_revision(156906, " ")
    def same(_payload, **kwargs):
        return deepcopy(review_env["explanation"])
    with pytest.raises(human.HumanReviewError, match="identical"):
        human.request_revision(156906, "Improve", explanation_runner=same, critic_runner=lambda p: None)
    assert human.get_review(156906)["current_version"] == 0


def test_new_pipeline_run_cannot_inherit_old_human_approval(review_env):
    human.approve_review(156906)
    path = orch.OUTPUT_DIR / "156906" / "patient_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["review_cycle"]["run_id"] = "run_beta"
    orch.save_json(path, state)
    fresh = human.get_review(156906)
    assert fresh["status"] == "pending"
    assert fresh["approved_version"] is None
    assert fresh["run_id"] == "run_beta"


def test_missing_state_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "OUTPUT_DIR", tmp_path)
    with pytest.raises(human.HumanReviewError, match="Required file"):
        human.get_review(156906)


def test_bad_patient_id_rejected():
    for bad in (True, -4, 0, "156906"):
        with pytest.raises(human.HumanReviewError):
            human.get_review(bad)


def test_reviewer_with_no_name_rejected(review_env):
    with pytest.raises(human.HumanReviewError, match="reviewer"):
        human.approve_review(156906, "  ")


def test_no_approval_if_critic_review_error(review_env):
    path = orch.OUTPUT_DIR / "156906" / "patient_state.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["critic"].update(status="review_error", approved=False)
    orch.save_json(path, data)
    with pytest.raises(human.HumanReviewError, match="not approved"):
        human.approve_review(156906)


def test_terminal_menu_approve(review_env, monkeypatch, capsys):
    from scripts import researcher_review
    monkeypatch.setattr("sys.argv", ["researcher_review", "156906", "--reviewer", "tester"])
    answers = iter(["3", "1", "y", "4", "7"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    researcher_review.main()
    printed = capsys.readouterr().out
    assert "Mortality Summary" in printed
    assert "Investigator approved version 0" in printed
    assert "Decision: approved" in printed
    assert human.get_review(156906)["status"] == "approved"


def test_terminal_menu_exit_without_decision(review_env, monkeypatch):
    from scripts import researcher_review
    monkeypatch.setattr("sys.argv", ["researcher_review", "156906"])
    monkeypatch.setattr("builtins.input", lambda prompt="": "7")
    researcher_review.main()
    assert human.get_review(156906)["status"] == "pending"


def test_cli_displays_critic_issues_after_human_revision(review_env, monkeypatch, capsys):
    from scripts import researcher_review

    def fake_revision(patient_id, feedback, reviewer):
        session = human.get_review(patient_id)
        new = deepcopy(session["versions"][-1])
        new["version"] = 1
        new["critic"] = {
            "status": "needs_revision", "approved": False,
            "issues": [{"code": "unsupported_treeshap_label", "severity": "minor",
                        "claim": "TreeSHAP explains...", "reason": "Method absent from reviewer evidence",
                        "evidence": "shap_explanation.method", "suggestion": "Confirm the method"}],
        }
        session["versions"].append(new)
        session["current_version"] = 1
        return session

    monkeypatch.setattr(researcher_review, "request_revision", fake_revision)
    monkeypatch.setattr("sys.argv", ["researcher_review", "156906"])
    answers = iter(["2", "Simplify SHAP", "", "7"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    researcher_review.main()
    output = capsys.readouterr().out
    assert "identified 1 issue(s)" in output
    assert "unsupported_treeshap_label" in output
    assert "Reason: Method absent from reviewer evidence" in output
    assert "Evidence: shap_explanation.method" in output
    assert "Suggestion: Confirm the method" in output


def test_cli_displays_revised_report_and_full_critic_prefix(review_env, monkeypatch, capsys):
    from scripts import researcher_review

    def fake_revision(patient_id, feedback, reviewer):
        session = deepcopy(human.get_review(patient_id))
        new = deepcopy(session["versions"][-1])
        new["version"] = 1
        new["explanation"]["explanation"]["shap_summary"] = "Clear revised SHAP explanation for researchers."
        new["critic"] = {"status": "approved", "approved": True, "issues": []}
        session["versions"].append(new)
        session["current_version"] = 1
        return session

    monkeypatch.setattr(researcher_review, "request_revision", fake_revision)
    monkeypatch.setattr("sys.argv", ["researcher_review", "156906"])
    answers = iter(["2", "Simplify", "", "7"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    researcher_review.main()
    output = capsys.readouterr().out
    assert "REVISED EXPLANATION — VERSION 1" in output
    assert "Clear revised SHAP explanation for researchers." in output
    assert "[CRITIC AGENT] Approved — No issues identified." in output
    assert "Awaiting explicit researcher approval" in output


def test_cli_confirmation_rejects_default_and_accepts_yes(review_env, monkeypatch, capsys):
    from scripts import researcher_review
    monkeypatch.setattr("sys.argv", ["researcher_review", "156906"])
    answers = iter(["1", "n", "1", "y", "7"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    researcher_review.main()
    output = capsys.readouterr().out
    assert "[CANCELLED] No approval was recorded." in output
    assert "Investigator approved version 0" in output
    session = human.get_review(156906)
    assert session["status"] == "approved"
    assert len([d for d in session["decisions"] if d["type"] == "approved"]) == 1


def test_cli_rejects_approval_when_critic_has_issues(review_env, monkeypatch, capsys):
    from scripts import researcher_review
    path = orch.OUTPUT_DIR / "156906" / "patient_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["critic"].update(status="needs_revision", approved=False, issues=[{"code": "test_issue", "severity": "minor"}])
    orch.save_json(path, state)
    monkeypatch.setattr("sys.argv", ["researcher_review", "156906"])
    # Avoid an endless loop by consuming one attempted approval and then exit.
    choices = iter(["1", "7"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(choices))
    researcher_review.main()
    output = capsys.readouterr().out
    assert "Cannot approve: Critic Agent has not approved" in output
    assert "test_issue" in output
    assert human.get_review(156906)["status"] == "pending"


def test_human_revision_sends_feedback_to_critic(review_env):
    inputs = []
    def fake_explanation(_payload, **_kwargs):
        revised = deepcopy(review_env['explanation'])
        revised['explanation']['shap_summary'] = 'The last respiratory rate changed model log-odds.'
        return revised
    def fake_critic(payload):
        inputs.append(payload)
        return {'agent':'critic', 'patientunitstayid':'156906', 'status':'approved',
                'approved':True, 'issues':[], 'feedback_compliance':{
                    'status':'satisfied', 'requirements':[], 'summary':'Met.', 'error_type':None}}
    session = human.request_revision(156906, 'Use plain SHAP language',
                                     explanation_runner=fake_explanation, critic_runner=fake_critic)
    assert inputs[0]['human_feedback'] == 'Use plain SHAP language'
    assert inputs[0]['previous_explanation'] == review_env['explanation']['explanation']
    assert inputs[0]['explanation_report'] == session['versions'][1]['explanation']['explanation']
    assert human.get_review(156906)['versions'][1]['critic']['feedback_compliance']['status'] == 'satisfied'


def test_cli_accessible_and_technical_views_are_separate(review_env, capsys):
    from scripts.researcher_review import show_accessible_report, show_technical_report
    version = deepcopy(human.get_review(156906)['versions'][0])
    version['explanation']['shap_explanation'] = {'method':'TreeSHAP','unit':'log-odds',
       'top_features':[{'feature':'rr_last','original_value':33.0,'shap_value':0.4086771607398987,
                        'direction':'increases','was_imputed':False}]}
    version['explanation']['mortality_prediction'] = {'probability':0.02536257542669773}
    original = deepcopy(version)
    show_accessible_report(version)
    accessible = capsys.readouterr().out
    assert 'last recorded respiratory rate' in accessible
    assert '--- What Influenced the Model? ---' in accessible
    assert 'log-odds' not in accessible
    assert 'rr_last' not in accessible
    assert '2.54%' in accessible
    assert '0.4086771607398987' not in accessible
    show_technical_report(version)
    technical = capsys.readouterr().out
    assert '0.4086771607398987' in technical
    assert '--- Model Prediction (raw values) ---' in technical
    assert '--- Mortality Summary ---' not in technical
    assert '--- Clinical Summary ---' not in technical
    assert version == original


def test_cli_displays_feedback_compliance_separately(capsys):
    from scripts.researcher_review import show_feedback_compliance
    show_feedback_compliance({'status':'approved','feedback_compliance':{
        'status':'partially_satisfied', 'summary':'One request remains.',
        'requirements':[{'requirement':'Use friendly names', 'status':'not_satisfied',
                         'evidence':'rr_last remains', 'suggestion':'Use respiratory rate'}]}})
    out = capsys.readouterr().out
    assert '[CRITIC AGENT — HUMAN FEEDBACK] partially_satisfied' in out
    assert 'Use respiratory rate' in out
    assert 'not a factual or clinical verdict' in out


def test_cli_menu_uses_separate_full_technical_accessible_views(review_env, monkeypatch, capsys):
    from scripts import researcher_review
    monkeypatch.setattr("sys.argv", ["researcher_review", "156906"])
    choices = iter(["3", "5", "6", "7"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(choices))
    researcher_review.main()
    output = capsys.readouterr().out
    assert "3. View full report" in output
    assert "4. View history" in output
    assert "5. View technical details" in output
    assert "6. View accessible explanation" in output
    assert "7. Exit" in output
    assert "--- Mortality Summary ---" in output
    assert "--- SHAP Feature Values (full precision) ---" in output
    assert "--- At a Glance ---" in output
    assert "[EXIT] Session saved" in output


def test_cli_invalid_choice_shows_correct_range(review_env, monkeypatch, capsys):
    from scripts import researcher_review
    monkeypatch.setattr("sys.argv", ["researcher_review", "156906"])
    choices = iter(["8", "7"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(choices))
    researcher_review.main()
    assert "Invalid option; enter 1–7." in capsys.readouterr().out


def test_accessible_view_keeps_imputation_and_limitations(review_env, capsys):
    from scripts.researcher_review import show_accessible_report
    version = deepcopy(human.get_review(156906)['versions'][0])
    version['explanation']['mortality_prediction'] = {'probability': 0.02536257542669773}
    version['explanation']['shap_explanation'] = {'top_features': [
        {'feature': 'BUN_last', 'original_value': None, 'shap_value': 0.16639505326747894,
         'direction': 'increases', 'was_imputed': True},
        {'feature': 'rr_last', 'original_value': 33.0, 'shap_value': 0.4086771607398987,
         'direction': 'increases', 'was_imputed': False},
    ]}
    show_accessible_report(version)
    output = capsys.readouterr().out
    assert 'model-filled value; not a measurement' in output
    assert 'recorded value: 33.0' in output
    assert '--- Important Limitations ---' in output
    assert 'Not clinically validated.' in output
    assert '0.4086771607398987' not in output


def test_technical_view_shows_structured_comparison_without_narrative(review_env, capsys):
    from scripts.researcher_review import show_technical_report
    version = deepcopy(human.get_review(156906)['versions'][0])
    version['explanation']['apache_comparison'] = {'versions': {'IV': {
        'apache_score': 18, 'apache_probability': 0.0068094464700123,
        'difference_percentage_points': 1.8553128956685432}},
        'note': 'Descriptive only.'}
    version['explanation']['shap_explanation'] = {'method': 'TreeSHAP',
        'unit': 'log-odds', 'all_features': [{'feature':'rr_last',
        'shap_value':0.4086771607398987, 'was_imputed':False,
        'original_value':33.0, 'processed_value':33.0, 'direction':'increases'}]}
    show_technical_report(version)
    output = capsys.readouterr().out
    assert '0.0068094464700123' in output
    assert 'SHAP=0.4086771607398987' in output
    assert 'processed=33.0' in output
    assert '--- Clinical Summary ---' not in output
    assert '--- Overall Explanation ---' not in output
