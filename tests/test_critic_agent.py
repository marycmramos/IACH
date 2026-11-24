import copy
import json
from pathlib import Path

import pytest

from agents.critic.critic_agent import run_critic_agent, validate_llm_review, build_evidence

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'severity.json'


@pytest.fixture
def severity():
    return json.loads(FIXTURE.read_text(encoding='utf-8'))


def good_llm(_messages):
    return {'issues': [], 'summary': 'No unsupported claims identified.'}


def invoke(severity, text, llm=good_llm):
    return run_critic_agent({'severity_result': severity, 'explanation_text': text}, llm_client=llm)


def test_correct_explanation_approved_with_successful_review(severity):
    result = invoke(severity, 'The XGBoost model mortality risk is 2.54%. SHAP describes model behavior, not causality.')
    assert result['status'] == 'approved'
    assert result['approved'] and result['issues'] == []


def test_wrong_xgboost_percentage(severity):
    result = invoke(severity, 'The XGBoost model mortality risk is 25.4%.')
    assert result['status'] == 'needs_revision'
    assert any(x['code'] == 'RISK_MISMATCH' for x in result['issues'])


def test_apache_percentage_not_confused_with_xgboost(severity):
    result = invoke(severity, 'APACHE IV mortality probability is 0.68%. The XGBoost model mortality risk is 2.54%.')
    assert result['status'] == 'approved'


def test_imputed_feature_presented_as_measured(severity):
    result = invoke(severity, 'The measured BUN_max was 21.')
    assert result['status'] == 'needs_revision'
    assert any(x['code'] == 'IMPUTED_AS_MEASURED' for x in result['issues'])


def test_real_measured_feature_not_flagged(severity):
    result = invoke(severity, 'The measured rr_last was 33.')
    assert result['status'] == 'approved'


def test_semantic_unsupported_claim_from_llm(severity):
    def fake_client(messages):
        assert 'explanation_to_review' in messages[1]['content']
        return {'issues': [{
            'code': 'UNSUPPORTED_DIAGNOSIS', 'category': 'support', 'severity': 'major',
            'claim': 'The patient has sepsis.', 'reason': 'No sepsis diagnosis is supplied.',
            'evidence': 'agent_context.clinical.raw_data',
            'suggestion': 'Remove the unsupported diagnosis.',
        }], 'summary': 'Unsupported clinical claim.'}
    result = invoke(severity, 'The patient has sepsis.', fake_client)
    assert result['status'] == 'needs_revision'
    assert result['issues'][0]['category'] == 'support'


def test_llm_failure_is_not_approval(severity):
    def broken(_messages):
        raise RuntimeError('offline')
    result = invoke(severity, 'The XGBoost model mortality risk is 2.54%.', broken)
    assert result['status'] == 'review_error' and not result['approved']
    assert result['llm_check'] == 'error'


def test_missing_llm_is_not_approval(severity):
    result = invoke(severity, 'The XGBoost model mortality risk is 2.54%.', None)
    assert result['status'] == 'review_error'
    assert result['llm_check'] == 'not_run'


def test_deterministic_findings_survive_llm_error(severity):
    result = invoke(severity, 'The XGBoost model mortality risk is 25.4%.', None)
    assert result['status'] == 'review_error'
    assert any(x['code'] == 'RISK_MISMATCH' for x in result['issues'])


def test_patient_id_mismatch_rejected(severity):
    with pytest.raises(ValueError, match='Patient ID mismatch'):
        run_critic_agent({'patientunitstayid': 111, 'severity_result': severity,
                          'explanation_text': 'Report'}, llm_client=good_llm)


def test_empty_report_rejected(severity):
    with pytest.raises(ValueError, match='nonempty'):
        invoke(severity, '  ')


def test_invalid_llm_json_rejected(severity):
    result = invoke(severity, 'Report text', lambda _: '{"issues":[], "summary":"ok", "extra":1}')
    assert result['status'] == 'review_error'
    assert result['error_type'] == 'ValidationError'


def test_no_mutation_of_input(severity):
    original = copy.deepcopy(severity)
    invoke(severity, 'The measured BUN_max was 21.')
    assert severity == original


def test_evidence_contains_original_data_without_id(severity):
    evidence = build_evidence(severity['result'])
    assert evidence['agent_context']['vital']['raw_data']['rr_last'] == 33.0
    assert evidence['agent_context']['laboratory']['raw_data']['BUN_max'] is None
    assert 'patientunitstayid' not in evidence['agent_context']['clinical']['raw_data']


def test_fenced_llm_json_accepted():
    review = validate_llm_review('```json\n{"issues": [], "summary":"ok"}\n```')
    assert review.issues == []

# Extra regression tests: different episode IDs and data completeness, deduplication,
# missing/invalid LLM replies, and unsupported clinical claims in mock reviews.

@pytest.mark.parametrize('episode_id, probability, missing', [
    (141765, 0.0342, 0),
    (222222, 0.31, 30),
    (333333, 0.003, 53),
])
def test_synthetic_episodes_are_not_hardcoded(severity, episode_id, probability, missing):
    case = copy.deepcopy(severity)
    case['patientunitstayid'] = episode_id
    case['result']['patientunitstayid'] = str(episode_id)
    case['result']['mortality_prediction']['probability'] = probability
    case['result']['data_quality']['n_missing_features'] = missing
    correct = f'The XGBoost model mortality risk is {probability * 100:.2f}%.'
    ok = invoke(case, correct)
    assert ok['patientunitstayid'] == str(episode_id)
    assert ok['status'] == 'approved'
    wrong = invoke(case, 'The XGBoost model mortality risk is 99%.')
    assert wrong['status'] == 'needs_revision'
    assert any(x['code'] == 'RISK_MISMATCH' for x in wrong['issues'])


def test_semantic_dedup_merges_realistic_llm_issues(severity):
    def fake(_messages):
        return {
            'issues': [
                {'code': 'INCORRECT_RISK_PERCENT', 'category': 'factual',
                 'severity': 'major', 'claim': 'The XGBoost model mortality risk is 25.4%.',
                 'reason': 'The correct value is 2.54%.',
                 'evidence': 'XGBoost probability = 0.02536257542669773',
                 'suggestion': 'Replace 25.4% with about 2.54%.'},
                {'code': 'IMPUTED_BUN_REPORTED_AS_MEASURED', 'category': 'factual',
                 'severity': 'major', 'claim': 'The measured BUN_max was 21.',
                 'reason': 'It was imputed, not measured.',
                 'evidence': 'BUN_max: original_value null; was_imputed true',
                 'suggestion': 'State BUN_max was imputed.'},
                {'code': 'CAUSAL_SHAP_INTERPRETATION', 'category': 'safety',
                 'severity': 'major', 'claim': 'Respiratory rate caused risk.',
                 'reason': 'SHAP cannot establish causality.',
                 'evidence': 'shap_explanation.unit=log_odds',
                 'suggestion': 'Describe model contribution instead.'},
                {'code': 'UNSUPPORTED_MODEL_SUPERIORITY', 'category': 'support',
                 'severity': 'critical', 'claim': 'XGBoost is superior to APACHE IV.',
                 'reason': 'Descriptive estimates do not establish superiority.',
                 'evidence': 'apache_comparison.status=descriptive',
                 'suggestion': 'Remove clinical superiority claim.'},
            ],
            'summary': 'Four distinct errors.',
        }
    result = invoke(severity, 'The XGBoost model mortality risk is 25.4%. '
                    'The measured BUN_max was 21. Respiratory rate caused risk. '
                    'XGBoost is superior to APACHE IV.', fake)
    assert result['status'] == 'needs_revision'
    assert len(result['issues']) == 4
    codes = {i['code'] for i in result['issues']}
    assert codes == {'RISK_MISMATCH', 'IMPUTED_AS_MEASURED',
                     'CAUSAL_SHAP_INTERPRETATION', 'UNSUPPORTED_MODEL_SUPERIORITY'}
    assert 'Additional review' in result['issues'][0]['evidence']


def test_different_imputed_features_are_not_collapsed(severity):
    report = 'The measured BUN_max was 21. The measured BUN_last was 20.'
    result = invoke(severity, report)
    assert result['status'] == 'needs_revision'
    assert len(result['issues']) == 2
    assert all(i['code'] == 'IMPUTED_AS_MEASURED' for i in result['issues'])


def test_duplicate_llm_issue_is_merged(severity):
    entry = {'code': 'UNSUPPORTED_DIAGNOSIS', 'category': 'support', 'severity': 'major',
             'claim': 'The patient has sepsis.', 'reason': 'No evidence for sepsis.',
             'evidence': 'clinical.raw_data', 'suggestion': 'Remove claim.'}
    result = invoke(severity, 'The patient has sepsis.',
                    lambda _: {'issues': [entry, entry], 'summary': 'Needs revision'})
    assert len(result['issues']) == 1


@pytest.mark.parametrize('bad_output', [
    '', 'not json', '{"issues": [], "summary": 2}',
    {'issues': [{'code': 'x'}], 'summary': 'bad'},
])
def test_malformed_llm_reply_never_approves(severity, bad_output):
    result = invoke(severity, 'The XGBoost model mortality risk is 2.54%.',
                    lambda _: bad_output)
    assert result['status'] == 'review_error'
    assert not result['approved']
    assert result['llm_check'] == 'error'


def test_llm_failure_preserves_multiple_deterministic_issues(severity):
    text = 'The XGBoost model mortality risk is 25.4%. The measured BUN_max was 21.'
    result = invoke(severity, text, lambda _: (_ for _ in ()).throw(RuntimeError('offline')))
    assert result['status'] == 'review_error'
    assert {i['code'] for i in result['issues']} == {'RISK_MISMATCH', 'IMPUTED_AS_MEASURED'}


def test_unsupported_claim_is_not_approved_with_mock_review(severity):
    claim = 'The patient must receive antibiotics immediately.'
    def semantic_client(_):
        return {'issues': [{
            'code': 'UNSUPPORTED_TREATMENT', 'category': 'safety', 'severity': 'critical',
            'claim': claim, 'reason': 'No evidence supports prescribing treatment.',
            'evidence': 'agent_context', 'suggestion': 'Remove treatment recommendation.'
        }], 'summary': 'Unsupported treatment recommendation.'}
    result = invoke(severity, claim, semantic_client)
    assert result['status'] == 'needs_revision'
    assert result['issues'][0]['category'] == 'safety'


def test_dedup_exact_live_iaedu_aliases(severity):
    """Regression: aliases observed in the actual November IAedu response."""
    def live_style(_messages):
        return {'issues': [
            {'code': 'incorrect_probability_scale', 'category': 'factual',
             'severity': 'critical', 'claim': 'The XGBoost model mortality risk is 25.4%.',
             'reason': 'Correct model estimate is 2.54%.',
             'evidence': 'mortality_prediction.probability = 0.02536257542669773',
             'suggestion': 'Use about 2.54%.'},
            {'code': 'imputed_value_reported_as_measured', 'category': 'factual',
             'severity': 'major', 'claim': 'The measured BUN_max was 21.',
             'reason': 'BUN_max was imputed.',
             'evidence': 'BUN_max: was_imputed=true; original_value=null',
             'suggestion': 'Do not call it measured.'},
            {'code': 'causal_interpretation_of_shap', 'category': 'safety',
             'severity': 'major', 'claim': 'Respiratory rate caused risk.',
             'reason': 'SHAP is not causal.', 'evidence': 'shap_explanation.unit=log_odds',
             'suggestion': 'Use contribution language.'},
            {'code': 'unsupported_model_superiority', 'category': 'support',
             'severity': 'critical', 'claim': 'XGBoost is superior to APACHE IV.',
             'reason': 'Only descriptive evidence available.',
             'evidence': 'apache_comparison.status=descriptive',
             'suggestion': 'Remove superiority.'},
        ], 'summary': 'Four distinct errors.'}
    report = ('The XGBoost model mortality risk is 25.4%. '
              'The measured BUN_max was 21. Respiratory rate caused risk. '
              'XGBoost is superior to APACHE IV.')
    result = invoke(severity, report, live_style)
    assert result['status'] == 'needs_revision'
    assert len(result['issues']) == 4
    assert {i['code'] for i in result['issues']} == {
        'RISK_MISMATCH', 'IMPUTED_AS_MEASURED',
        'causal_interpretation_of_shap', 'unsupported_model_superiority'}
    risk = next(i for i in result['issues'] if i['code'] == 'RISK_MISMATCH')
    assert 'Additional review' in risk['evidence']


def test_new_aliases_do_not_merge_distinct_imputed_features(severity):
    def fake(_messages):
        return {'issues': [
            {'code': 'imputed_value_reported_as_measured', 'category': 'factual',
             'severity': 'major', 'claim': f'The measured {name} was recorded.',
             'reason': 'Imputed input.', 'evidence': f'{name}: was_imputed=true',
             'suggestion': 'Mark as imputed.'}
            for name in ('BUN_max', 'BUN_last')], 'summary': 'Two errors.'}
    report = 'The measured BUN_max was 21. The measured BUN_last was 20.'
    result = invoke(severity, report, fake)
    assert len(result['issues']) == 2
    assert all(i['code'] == 'IMPUTED_AS_MEASURED' for i in result['issues'])


def test_new_live_probability_code_merges_without_alias(severity):
    """The new IAedu code varies on each run; the target/evidence remains stable."""
    def llm(_):
        return {'issues': [{
            'code': 'INCORRECT_MORTALITY_PERCENT', 'category': 'factual',
            'severity': 'major', 'claim': 'The XGBoost model mortality risk is 25.4%.',
            'reason': 'The reported model probability is 0.02536257542669773, '
                      'which corresponds to about 2.54%, not 25.4%.',
            'evidence': 'mortality_prediction.probability = 0.02536257542669773',
            'suggestion': 'Use approximately 2.54%.'
        }], 'summary': 'Wrong probability.'}
    result = invoke(severity, 'The XGBoost model mortality risk is 25.4%.', llm)
    assert len(result['issues']) == 1
    assert result['issues'][0]['code'] == 'RISK_MISMATCH'
    assert 'Additional review' in result['issues'][0]['evidence']


def test_arbitrary_code_still_merges_same_probability(severity):
    def llm(_):
        return {'issues': [{
            'code': 'any_new_name_123', 'category': 'factual', 'severity': 'major',
            'claim': 'The XGBoost model mortality risk is 25.4%.',
            'reason': 'This is incorrect: the model probability is 0.02536.',
            'evidence': 'mortality_prediction.probability=0.02536257542669773',
            'suggestion': 'Correct the figure.'
        }], 'summary': 'Wrong probability.'}
    result = invoke(severity, 'The XGBoost model mortality risk is 25.4%.', llm)
    assert len(result['issues']) == 1


def test_same_llm_code_for_different_imputed_features_is_not_merged(severity):
    def llm(_):
        return {'issues': [{
            'code': 'unexpected_code', 'category': 'factual', 'severity': 'major',
            'claim': f'The measured {feature} was recorded.',
            'reason': 'This was imputed, not measured.',
            'evidence': f'{feature}: was_imputed=true',
            'suggestion': 'Disclose imputation.'
        } for feature in ('BUN_max', 'BUN_last')], 'summary': 'Two separate values.'}
    result = invoke(severity, 'The measured BUN_max was 21. The measured BUN_last was 20.', llm)
    assert len(result['issues']) == 2
    assert all(issue['code'] == 'IMPUTED_AS_MEASURED' for issue in result['issues'])


def test_apache_probability_not_deduplicated_with_xgboost(severity):
    def llm(_):
        return {'issues': [{
            'code': 'any_new_name_123', 'category': 'factual', 'severity': 'major',
            'claim': 'APACHE IV mortality risk is 25.4%.',
            'reason': 'Incorrect APACHE number.',
            'evidence': 'apache_comparison.versions.IV.apache_probability',
            'suggestion': 'Correct APACHE IV number.'
        }], 'summary': 'Separate issue.'}
    result = invoke(severity, 'The XGBoost model mortality risk is 25.4%. '
                     'APACHE IV mortality risk is 25.4%.', llm)
    assert len(result['issues']) == 2


def test_llm_issue_with_two_imputed_features_preserved(severity):
    def llm(_):
        return {'issues': [{
            'code': 'unexpected_code', 'category': 'factual', 'severity': 'major',
            'claim': 'The measured BUN_max and BUN_last were recorded.',
            'reason': 'Both were imputed.',
            'evidence': 'BUN_max and BUN_last: was_imputed=true',
            'suggestion': 'Disclose both missing measurements.'
        }], 'summary': 'Compound assertion.'}
    result = invoke(severity, 'The measured BUN_max and BUN_last were recorded.', llm)
    # Ambiguous multi-feature claims are preserved rather than collapsed blindly.
    assert len(result['issues']) == 2
    assert any(i['claim'].startswith('The measured BUN_max') for i in result['issues'])


def test_treeshap_method_is_sent_to_semantic_reviewer(severity):
    """Regression for the observed false-positive unsupported_treeshap_label."""
    def inspecting_client(messages):
        import json
        payload = json.loads(messages[1]["content"])
        evidence = payload["evidence"]["shap_explanation"]
        assert evidence["method"] == "TreeSHAP"
        assert evidence["unit"] == "log_odds"
        assert "TreeSHAP" in payload["explanation_to_review"]
        return {"issues": [], "summary": "TreeSHAP is supported by method metadata."}

    result = invoke(severity, "TreeSHAP explains feature contributions in log-odds.", inspecting_client)
    assert result["status"] == "approved"


def test_missing_shap_method_is_not_invented(severity):
    copy_of_severity = copy.deepcopy(severity["result"])
    copy_of_severity["shap_explanation"].pop("method", None)
    assert build_evidence(copy_of_severity)["shap_explanation"]["method"] is None
