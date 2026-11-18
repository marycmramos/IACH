"""Unit tests for Severity Agent. No IAedu requests or patient CSVs."""
from copy import deepcopy
import json

import pytest

from agents.severity import severity_agent as severity


@pytest.fixture
def fake_prediction(monkeypatch):
    """Replace inference with a deterministic prediction (no model file needed)."""
    prediction = {
        "mortality_probability": 0.034,
        "outcome": "icu_mortality_after_24h",
        "base_probability": 0.05,
        "shap_unit": "log_odds",
        "shap_values": [
            {"feature": "rr_last", "original_value": 37.0,
             "was_imputed": False, "shap_value": 0.4, "direction": "aumenta"},
            {"feature": "BUN_max", "original_value": None,
             "was_imputed": True, "shap_value": -0.2, "direction": "diminui"},
        ],
        "top_shap_features": [
            {"feature": "rr_last", "original_value": 37.0,
             "was_imputed": False, "shap_value": 0.4, "direction": "increases"},
        ],
        "n_features": 53,
        "n_missing_features": 1,
        "missing_features": ["BUN_max"],
        "warnings": [],
    }
    monkeypatch.setattr(severity, "predict_mortality", lambda features: deepcopy(prediction))
    return prediction


def _input(apache=None):
    return {
        "patientunitstayid": 141765,
        "features": {"rr_last": 37.0, "BUN_max": None},
        "agent_results": {},
        "apache_result": apache,
    }


def _apache():
    return {
        "IV": {"version": "IV", "score": 47, "mortality_probability": 0.008},
        "IVa": {"version": "IVa", "score": 47, "mortality_probability": 0.012},
    }


def test_apache_both_versions():
    comparison = severity.compare_apache(0.034, _apache())
    assert comparison["status"] == "descriptive"
    assert set(comparison["versions"]) == {"IV", "IVa"}
    assert comparison["versions"]["IV"]["difference_percentage_points"] == pytest.approx(2.6)
    assert comparison["versions"]["IVa"]["difference_percentage_points"] == pytest.approx(2.2)
    assert comparison["versions"]["IV"]["apache_score"] == 47


def test_apache_missing_or_empty():
    for missing in (None, {}):
        comparison = severity.compare_apache(0.034, missing)
        assert comparison["status"] == "unavailable"
        assert comparison["versions"] == {}


def test_apache_single_version():
    comparison = severity.compare_apache(0.034, {"IV": _apache()["IV"]})
    assert comparison["status"] == "descriptive"
    assert set(comparison["versions"]) == {"IV"}


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), True, "0.5"])
def test_apache_rejects_invalid_probabilities(bad):
    with pytest.raises(ValueError):
        severity.compare_apache(0.034, {"IV": {"mortality_probability": bad}})


def test_apache_rejects_unknown_version():
    with pytest.raises(ValueError):
        severity.compare_apache(0.034, {"III": {"mortality_probability": 0.02}})


def test_severity_without_apache_and_llm(fake_prediction):
    result = severity.run_severity_agent(_input(), llm_client=None)
    assert result["status"] == "success"
    assert result["mortality_prediction"]["probability"] == pytest.approx(0.034)
    assert result["apache_comparison"]["status"] == "unavailable"
    assert result["llm_analysis"]["status"] == "not_run"
    assert result["data_quality"]["n_missing_features"] == 1
    assert result["shap_explanation"]["all_features"][0]["direction"] == "increases"
    assert result["shap_explanation"]["all_features"][1]["direction"] == "decreases"


def test_severity_with_apache_and_mock_llm(fake_prediction):
    seen = []

    def fake_llm(messages):
        seen.extend(messages)
        evidence = json.loads(messages[1]["content"])
        assert evidence["apache_comparison"]["status"] == "descriptive"
        assert evidence["prediction"]["n_missing_features"] == 1
        assert "patientunitstayid" not in messages[1]["content"]
        return {
            "findings": ["Observed evidence only."],
            "shap_interpretation": ["SHAP is in log-odds."],
            "apache_assessment": "Separate descriptive APACHE estimates.",
            "limitations": ["Academic prototype."],
        }

    result = severity.run_severity_agent(_input(_apache()), llm_client=fake_llm)
    assert result["status"] == "success"
    assert result["apache_comparison"]["status"] == "descriptive"
    assert result["llm_analysis"]["status"] == "success"
    assert len(seen) == 2
    assert result["mortality_prediction"]["probability"] == pytest.approx(0.034)


def test_llm_failure_does_not_block_prediction(fake_prediction):
    def fail(_messages):
        raise RuntimeError("IAedu unavailable")

    result = severity.run_severity_agent(_input(), llm_client=fail)
    assert result["status"] == "success"
    assert result["llm_analysis"]["status"] == "error"
    assert result["mortality_prediction"]["probability"] == pytest.approx(0.034)


def test_invalid_severity_input():
    with pytest.raises(TypeError):
        severity.run_severity_agent({"patientunitstayid": 141765, "features": None}, llm_client=None)
    with pytest.raises(ValueError):
        severity.run_severity_agent({"features": {}}, llm_client=None)


# Additional cases: schema validation and IAedu SSE parsing, all offline.

def _assessment():
    return {"findings": ["Observed finding"],
            "shap_interpretation": ["rr_last increases log-odds"],
            "apache_assessment": None, "limitations": ["Not clinically validated"]}


def test_validate_llm_json_and_code_fence():
    valid = _assessment()
    assert severity.validate_llm_assessment(valid) == valid
    assert severity.validate_llm_assessment(json.dumps(valid)) == valid
    assert severity.validate_llm_assessment("```json\n" + json.dumps(valid) + "\n```") == valid


@pytest.mark.parametrize("bad", [
    "not JSON", "{}", {**_assessment(), "invented_field": "no"},
    {**_assessment(), "findings": "not a list"},
])
def test_validate_llm_rejects_invalid_schema(bad):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        severity.validate_llm_assessment(bad)


def test_llm_context_keeps_findings_not_raw_data():
    context = severity.llm_agent_context({
        "vital": {"status": "success", "raw_data": {"private": 42},
                  "agent_output": {"findings": [{"variable": "rr_last", "value": 37}]},
                  "patterns": [{"variable": "rr", "pattern_type": "observed_range"}]},
        "clinical": "invalid shape",
    })
    assert context["vital"]["status"] == "success"
    assert context["vital"]["findings"][0]["variable"] == "rr_last"
    assert "raw_data" not in context["vital"]
    assert context["clinical"]["status"] == "unrecognized_output"


def test_iaedu_adapter_builds_trusted_prompt(monkeypatch):
    recorded = []
    monkeypatch.setattr(severity, "ask_iaedu", lambda prompt: recorded.append(prompt) or json.dumps(_assessment()))
    messages = [{"role": "system", "content": "SAFE SYSTEM"},
                {"role": "user", "content": '{"x":1}'}]
    assert severity.iaedu_llm_client(messages) == _assessment()
    assert "SAFE SYSTEM" in recorded[0]
    assert '{"x":1}' in recorded[0]
    with pytest.raises(ValueError):
        severity.iaedu_llm_client(messages[:1])


class _FakeResponse:
    def __init__(self, lines, error=None):
        self.lines = lines
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        if self.error:
            raise self.error

    def iter_lines(self, decode_unicode=True):
        return iter(self.lines)


@pytest.fixture
def iaedu_env(monkeypatch):
    monkeypatch.setenv("IAEDU_API_ENDPOINT", "https://example.org/stream")
    monkeypatch.setenv("IAEDU_CHANNEL_ID", "test-channel")
    monkeypatch.setenv("IAEDU_API_KEY", "dummy-not-real")


def test_iaedu_sse_token_stream_without_network(iaedu_env, monkeypatch):
    captured = []
    lines = [': keepalive', 'event: message', 'data: {"type":"token","content":"Hel"}',
             'data: {"type":"token","content":"lo"}', 'data: [DONE]']
    def fake_post(url, **kwargs):
        captured.append((url, kwargs))
        return _FakeResponse(lines)
    monkeypatch.setattr(severity.requests, "post", fake_post)
    assert severity.ask_iaedu("synthetic query") == "Hello"
    assert captured[0][0] == "https://example.org/stream"
    assert captured[0][1]["stream"] is True
    assert captured[0][1]["files"]["message"] == (None, "synthetic query")


def test_iaedu_sse_final_message_takes_precedence(iaedu_env, monkeypatch):
    lines = ['data: {"type":"token","content":"partial"}',
             'data: {"type":"message","content":{"content":"final"}}', 'data: [DONE]']
    monkeypatch.setattr(severity.requests, "post", lambda *_args, **_kwargs: _FakeResponse(lines))
    assert severity.ask_iaedu("test") == "final"


@pytest.mark.parametrize("events, error_type", [
    (['data: {"type":"error","message":"upstream failed"}'], RuntimeError),
    (['data: {"type":"unknown"}', 'data: [DONE]'], ValueError),
])
def test_iaedu_sse_fails_safely(iaedu_env, monkeypatch, events, error_type):
    monkeypatch.setattr(severity.requests, "post", lambda *_args, **_kwargs: _FakeResponse(events))
    with pytest.raises(error_type):
        severity.ask_iaedu("test")


def test_iaedu_network_error_propagates(iaedu_env, monkeypatch):
    import requests
    def fail_post(*_args, **_kwargs):
        raise requests.Timeout("synthetic timeout")
    monkeypatch.setattr(severity.requests, "post", fail_post)
    with pytest.raises(requests.Timeout):
        severity.ask_iaedu("test")


def test_iaedu_missing_credentials_rejected(monkeypatch):
    monkeypatch.delenv("IAEDU_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="IAedu"):
        severity.ask_iaedu("test")


def test_iaedu_rejects_plain_http(iaedu_env, monkeypatch):
    monkeypatch.setenv("IAEDU_API_ENDPOINT", "http://example.org/stream")
    with pytest.raises(ValueError, match="HTTPS"):
        severity.ask_iaedu("test")


def test_llm_bad_schema_keeps_ml_result(fake_prediction):
    result = severity.run_severity_agent(_input(_apache()), llm_client=lambda _messages: {"invalid": 1})
    assert result["status"] == "success"
    assert result["llm_analysis"]["status"] == "error"
    assert result["llm_analysis"]["error_type"] == "ValidationError"
    assert result["mortality_prediction"]["probability"] == pytest.approx(0.034)


def test_rejects_unknown_agent_in_severity_input():
    bad = _input()
    bad["agent_results"] = {"unexpected": {"status": "success"}}
    with pytest.raises(ValueError, match="Unknown agent"):
        severity.run_severity_agent(bad, llm_client=None)
