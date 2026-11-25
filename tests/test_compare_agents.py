"""Offline tests for evaluation/compare_agents.py; no IAedu calls."""
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "compare_agents.py"
spec = importlib.util.spec_from_file_location("compare_agents", MODULE_PATH)
ca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ca)


def reference():
    return {
        "mortality_prediction": {"probability": 0.03417055681347847},
        "shap_explanation": {"top_features": [{"feature": "age", "was_imputed": False}]},
        "apache_comparison": {"status": "unavailable", "versions": {}},
    }


def result():
    return {"mortality_prediction": {"probability": 0.03417055681347847},
            "explanation": {"mortality_summary": "3.42%", "clinical_summary": "Evidence summary.",
                            "shap_summary": "age contributed +0.2 log-odds to the model prediction.",
                            "overall_explanation": "APACHE is unavailable.",
                            "limitations": ["Not clinically validated."]}}


def test_valid_report_has_zero_technical_warnings():
    checks = ca._inspect(result(), reference())
    assert checks["technical_issue_count"] == 0
    assert checks["shap_features_mentioned"] == 1


def test_wrong_probability_flagged():
    data = result()
    data["explanation"]["mortality_summary"] = "30.00%"
    assert not ca._inspect(data, reference())["mortality_ok"]


def test_shap_percentage_flagged():
    data = result()
    data["explanation"]["shap_summary"] = "SHAP age contributes 12% to mortality."
    assert not ca._inspect(data, reference())["shap_units_ok"]


def test_imputed_feature_requires_disclosure():
    ref = reference()
    ref["shap_explanation"]["top_features"][0]["was_imputed"] = True
    assert not ca._inspect(result(), ref)["imputation_disclosed"]
    data = result()
    data["explanation"]["limitations"].append("Age value was imputed.")
    assert ca._inspect(data, ref)["imputation_disclosed"]


def test_apache_descriptive_requires_probability_and_label():
    ref = reference()
    ref["apache_comparison"] = {
        "status": "descriptive", "versions": {
            "IV": {"status": "descriptive", "apache_probability": 0.008247191387781},
            "IVa": {"status": "descriptive", "apache_probability": 0.012311455455498},
        }
    }
    data = result()
    assert not ca._inspect(data, ref)["apache_ok"]
    data["explanation"]["overall_explanation"] = "APACHE IV 0.82%; APACHE IVa 1.23%."
    assert ca._inspect(data, ref)["apache_ok"]


def test_missing_sections_flagged():
    data = result()
    del data["explanation"]["clinical_summary"]
    assert not ca._inspect(data, reference())["sections_ok"]


def test_natural_language_names_are_recognized_for_all_ten_features():
    names = ["sbp_last", "sao2_min", "glucose_first", "potassium_last",
             "dbp_max", "BUN_max", "BUN_last", "sbp_min", "dbp_last", "rr_mean"]
    text = (
        "last systolic blood pressure 85, minimum oxygen saturation 72, "
        "first glucose 313, last potassium 6.2, maximum diastolic blood pressure 145, "
        "maximum BUN 93, last BUN 93, minimum systolic blood pressure 49, "
        "last diastolic blood pressure 55, mean respiratory rate (imputed)"
    )
    assert all(ca._feature_mentioned(name, text) for name in names)
    ref = reference()
    ref["shap_explanation"]["top_features"] = [
        {"feature": name, "was_imputed": name == "rr_mean"} for name in names
    ]
    data = result()
    data["explanation"]["shap_summary"] = text + " in log-odds."
    data["explanation"]["limitations"].append("Mean respiratory rate was imputed.")
    assert ca._inspect(data, ref)["shap_features_mentioned"] == 10


def test_feature_aliases_do_not_match_separated_words_or_wrong_statistics():
    assert not ca._feature_mentioned("sbp_last", "Last measurement was recorded. Systolic blood pressure varied.")
    assert not ca._feature_mentioned("sbp_last", "minimum systolic blood pressure was 49")
    assert not ca._feature_mentioned("rr_mean", "maximum respiratory rate was 39")
    assert not ca._feature_mentioned("BUN_last", "last glucose and maximum BUN were observed")


def test_canonical_names_and_unknown_features_still_work():
    assert ca._feature_mentioned("sbp_last", "The sbp_last was 85")
    assert ca._feature_mentioned("age", "age contributed positively")
    assert not ca._feature_mentioned("unknown_feature", "unknown feature was mentioned")
    assert ca._feature_mentioned("unknown_feature", "unknown_feature was mentioned")


def test_name_and_statistic_are_matched_in_both_orders():
    assert ca._feature_mentioned("BUN_max", "BUN maximum was 93")
    assert ca._feature_mentioned("rr_mean", "respiratory rate average was 20")
    assert ca._feature_mentioned("sao2_min", "lowest oxygen saturation was 72")

import json


def setup_three_arm_case(monkeypatch, tmp_path, approved=True, revisions=0):
    monkeypatch.setattr(ca, 'ROOT', tmp_path)
    monkeypatch.setattr(ca, 'PATIENTS', tmp_path / 'outputs' / 'patients')
    episode = '176730'
    folder = ca.PATIENTS / episode
    folder.mkdir(parents=True)
    sev = reference()
    if True:
        sev['patientunitstayid'] = int(episode)
    r = result()
    r['patientunitstayid'] = int(episode)
    for name, data in [('severity', {'status':'success', **sev}),
                       ('single_agent', {'status':'success', **r}),
                       ('explanation_B', {'status':'success', **r}),
                       ('explanation', {'status':'success', **r})]:
        (folder / f'{name}.json').write_text(json.dumps({'patientunitstayid': int(episode), 'result':data}))
    run_id = '20261125T112121440992Z_abcd1234'
    critic = {'patientunitstayid':str(episode), 'status':'approved' if approved else 'needs_revision', 'approved':approved, 'issues':[]}
    (folder/'critic.json').write_text(json.dumps({'patientunitstayid':int(episode),'result':critic}))
    history_path = f'outputs/patients/{episode}/revision_runs/{run_id}'
    history = [{'round':i,'critic_status': 'approved' if i == revisions and approved else 'needs_revision','issue_count':0} for i in range(revisions+1)]
    cycle = {'run_id':run_id,'history':history,'revisions_used':revisions,'stop_reason':'approved' if approved else 'revision_limit_reached','human_review_required':True,'history_path':history_path}
    state={'patientunitstayid':int(episode),'status':'success','critic':critic,'review_cycle':cycle}
    (folder/'patient_state.json').write_text(json.dumps(state))
    target=tmp_path/history_path
    target.mkdir(parents=True)
    (target / f'round_{revisions:02d}.json').write_text(json.dumps({'round':revisions,'critic':critic,'explanation':{'status':'success', **r}}))
    return episode, folder


def test_three_architectures_are_compared(monkeypatch, tmp_path):
    episode, _ = setup_three_arm_case(monkeypatch, tmp_path)
    rows = ca.compare(episode, 'test')
    assert [x['architecture'] for x in rows] == ['A_single', 'B_multi_no_critic', 'C_multi_with_critic']
    assert [x['technical_issue_count'] for x in rows] == [0,0,0]
    assert rows[2]['critic_status'] == 'approved'
    assert rows[2]['revisions_used'] == 0
    assert rows[0]['critic_status'] == ''


def test_multiple_revisions_are_reported(monkeypatch, tmp_path):
    episode, _ = setup_three_arm_case(monkeypatch, tmp_path, revisions=2)
    rows = ca.compare(episode, 'test')
    assert len(rows) == 3
    assert rows[2]['revisions_used'] == 2


def test_unapproved_critic_is_not_treated_as_c(monkeypatch, tmp_path):
    episode, _ = setup_three_arm_case(monkeypatch, tmp_path, approved=False)
    assert ca.compare(episode, 'test') == []


def test_missing_B_is_not_silently_replaced_with_C(monkeypatch, tmp_path):
    episode, folder = setup_three_arm_case(monkeypatch, tmp_path)
    (folder / 'explanation_B.json').unlink()
    assert ca.compare(episode, 'test') == []


def test_stale_or_mismatched_critic_history_is_rejected(monkeypatch, tmp_path):
    episode, folder = setup_three_arm_case(monkeypatch, tmp_path)
    state = json.loads((folder/'patient_state.json').read_text())
    state['review_cycle']['run_id']='some_other_run'
    (folder/'patient_state.json').write_text(json.dumps(state))
    assert ca.compare(episode, 'test') == []
