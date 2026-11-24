"""Tests for the one-command launch and non-mutating initial overview."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import types

from scripts import researcher_review as cli
from scripts import start_iach as start


def _session():
    return {
        'patientunitstayid': 176730, 'status': 'pending', 'current_version': 0,
        'versions': [{'version': 0,
                      'explanation': {'mortality_prediction': {'probability': 0.01117834},
                                      'shap_explanation': {'top_features': [
                                          {'feature': 'hr_mean', 'direction': 'increases', 'was_imputed': False},
                                          {'feature': 'BUN_last', 'direction': 'decreases', 'was_imputed': False}]},
                                      'explanation': {'limitations': ['Model not clinically validated.',
                                                                       'SHAP is not causal.']}},
                      'critic': {'status': 'approved', 'approved': True}}],
    }


def test_overview_reads_saved_data_without_mutating(capsys):
    sess = _session()
    original = deepcopy(sess)
    cli.show_initial_overview(sess)
    captured = capsys.readouterr().out
    assert '1.12%' in captured
    assert 'average heart rate' in captured
    assert 'last BUN' in captured
    assert 'SHAP explains model behavior' in captured
    assert sess == original


def test_resume_does_not_run_orchestrator(monkeypatch, tmp_path):
    (tmp_path / '176730').mkdir()
    (tmp_path / '176730' / 'patient_state.json').write_text('{}')
    orch = SimpleNamespace(OUTPUT_DIR=tmp_path, run_orchestrator=lambda _: (_ for _ in ()).throw(AssertionError('rerun')))
    monkeypatch.setitem(sys.modules, 'orchestration.orchestrator', orch)
    from orchestration import orchestrator
    monkeypatch.setattr(orchestrator, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(orchestrator, 'run_orchestrator', orch.run_orchestrator)
    calls = []
    monkeypatch.setattr(cli, 'run_review_cli', lambda *args, **kwargs: calls.append((args, kwargs)))
    assert start.launch(176730, resume=True)
    assert calls[0][0] == (176730,)
    assert calls[0][1]['show_overview'] is True


def test_incomplete_pipeline_does_not_open_review(monkeypatch, tmp_path):
    from orchestration import orchestrator
    monkeypatch.setattr(orchestrator, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(orchestrator, 'run_orchestrator', lambda _: {
        'status': 'partial', 'agents': {'vital': {'status': 'insufficient_data'}},
        'explanation': 'not_run', 'critic': 'not_run'})
    monkeypatch.setattr(cli, 'run_review_cli', lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError('open')))
    assert start.launch(209140) is False


def test_new_run_hides_detailed_prints_and_stores_log(monkeypatch, tmp_path, capsys):
    from orchestration import orchestrator
    monkeypatch.setattr(orchestrator, 'OUTPUT_DIR', tmp_path)

    def fake_pipeline(patient_id):
        print('Vital: (2520, 29) | C:/private/vital_agent_data.csv')
        print('IAedu carregada do .env (credenciais ocultas).')
        return {'agents': {'vital': {'status': 'success'}, 'laboratory': {'status': 'success'},
                           'clinical': {'status': 'success'}},
                'severity': {'status': 'success'}, 'explanation': {'status': 'success'},
                'critic': {'status': 'approved'}}

    monkeypatch.setattr(orchestrator, 'run_orchestrator', fake_pipeline)
    monkeypatch.setattr(cli, 'run_review_cli', lambda *_args, **_kwargs: None)
    assert start.launch(162841)
    display = capsys.readouterr().out
    assert 'Vital: (2520, 29)' not in display
    assert 'IAedu carregada' not in display
    assert '[OK] Critic review approved' in display
    logs = list((tmp_path / '162841' / 'logs').glob('pipeline_*.log'))
    assert len(logs) == 1
    assert 'Vital: (2520, 29)' in logs[0].read_text(encoding='utf-8')


def test_verbose_preserves_detailed_prints(monkeypatch, tmp_path, capsys):
    from orchestration import orchestrator
    monkeypatch.setattr(orchestrator, 'OUTPUT_DIR', tmp_path)

    def fake_pipeline(_):
        print('Detailed Orchestrator message')
        return {'agents': {}, 'explanation': {'status': 'success'},
                'critic': {'status': 'approved'}}

    monkeypatch.setattr(orchestrator, 'run_orchestrator', fake_pipeline)
    monkeypatch.setattr(cli, 'run_review_cli', lambda *_a, **_kw: None)
    assert start.launch(162841, verbose=True)
    assert 'Detailed Orchestrator message' in capsys.readouterr().out
    assert not (tmp_path / '162841' / 'logs').exists()


def test_failure_shows_short_message_and_keeps_log(monkeypatch, tmp_path, capsys):
    from orchestration import orchestrator
    monkeypatch.setattr(orchestrator, 'OUTPUT_DIR', tmp_path)

    def fake_pipeline(_):
        print('Vital: data incomplete')
        return {'status': 'partial', 'agents': {'vital': {'status': 'insufficient_data'}},
                'explanation': 'not_run', 'critic': 'not_run'}

    monkeypatch.setattr(orchestrator, 'run_orchestrator', fake_pipeline)
    monkeypatch.setattr(cli, 'run_review_cli', lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError()))
    assert not start.launch(209140)
    output = capsys.readouterr().out
    assert 'vital=insufficient_data' in output
    assert 'Pipeline log:' in output
    assert 'Vital: data incomplete' not in output
