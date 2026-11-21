"""Regression tests for optional human-feedback evaluation within Critic Agent."""
import json

from agents.critic.critic_agent import run_critic_agent


def severity():
    return {
        'agent': 'severity', 'status': 'success', 'patientunitstayid': '176730',
        'mortality_prediction': {'probability': 0.011178342625498772},
        'shap_explanation': {'unit': 'log_odds', 'method': 'TreeSHAP', 'all_features': []},
    }


def payload(with_feedback=True):
    result = {
        'patientunitstayid': 176730,
        'severity_result': severity(),
        'explanation_text': 'The model estimated an ICU mortality probability of 1.12%.',
    }
    if with_feedback:
        result.update({
            'human_feedback': 'Explain SHAP simply and use understandable variable names.',
            'explanation_report': {'shap_summary': 'Average heart rate increased the model output.'},
            'previous_explanation': {'shap_summary': 'hr_mean SHAP = +0.321.'},
        })
    return result


def test_feedback_satisfied_and_factual_approval_independent():
    calls = []

    def mock_llm(messages):
        calls.append(messages)
        if len(calls) == 1:
            return {'issues': [], 'summary': 'Factual review passed.'}
        return {'requirements': [{
            'requirement': 'Use understandable variable names',
            'status': 'satisfied', 'evidence': 'Average heart rate is present.',
            'suggestion': 'No change needed',
        }], 'summary': 'Request met.'}

    result = run_critic_agent(payload(), llm_client=mock_llm)
    assert result['status'] == 'approved'
    assert result['feedback_compliance']['status'] == 'satisfied'
    assert len(calls) == 2
    assert json.loads(calls[1][1]['content'])['human_feedback'] == payload()['human_feedback']


def test_partially_satisfied_does_not_override_factual_approval():
    count = 0

    def mock_llm(_messages):
        nonlocal count
        count += 1
        if count == 1:
            return {'issues': [], 'summary': 'Factually correct.'}
        return {'requirements': [
            {'requirement': 'Explain SHAP', 'status': 'satisfied', 'evidence': 'Explained.', 'suggestion': 'No change needed'},
            {'requirement': 'Readable names', 'status': 'not_satisfied', 'evidence': 'Some codes remain.', 'suggestion': 'Replace codes.'},
        ], 'summary': 'Partially satisfied.'}

    result = run_critic_agent(payload(), llm_client=mock_llm)
    assert result['approved'] is True
    assert result['feedback_compliance']['status'] == 'partially_satisfied'


def test_invalid_feedback_response_yields_review_error_without_leaking_response():
    count = 0

    def mock_llm(_messages):
        nonlocal count
        count += 1
        return {'issues': [], 'summary': 'Factual review passed.'} if count == 1 else 'THIS IS NOT JSON'

    result = run_critic_agent(payload(), llm_client=mock_llm)
    assert result['status'] == 'approved'
    assert result['feedback_compliance']['status'] == 'review_error'
    assert 'THIS IS NOT JSON' not in json.dumps(result)


def test_automatic_review_without_human_feedback_does_not_call_second_llm():
    calls = []

    def mock_llm(messages):
        calls.append(messages)
        return {'issues': [], 'summary': 'No issues.'}

    result = run_critic_agent(payload(with_feedback=False), llm_client=mock_llm)
    assert result['status'] == 'approved'
    assert result['feedback_compliance'] is None
    assert len(calls) == 1
