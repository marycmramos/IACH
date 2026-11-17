"""Testes do Orchestrator IACH (pytest).

Colocar este ficheiro ao lado de Orchestrator_imp.ipynb e executar:
    python -m pip install pytest nbformat nbclient ipykernel
    python -m pytest -v test_orchestrator.py

Não executa notebooks de agentes nem chama a IAedu. Os ficheiros JSON dos
 testes são guardados numa pasta temporária criada pelo pytest.
"""

import ast
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parent.parent
ORCHESTRATOR_NOTEBOOK = ROOT / "Orchestrator_imp.ipynb"

print("Pasta principal:", ROOT)
print("Notebook procurado:", ORCHESTRATOR_NOTEBOOK)
print("Existe?", ORCHESTRATOR_NOTEBOOK.exists())

# Estes são os blocos de implementação do notebook que queremos carregar.
# A célula de inspeção de agentes, os testes e a execução real são excluídos.
REQUIRED_FUNCTIONS = {
    "patient_folder",
    "agent_output_path",
    "notebook_fingerprint",
    "save_json",
    "load_cached_agent",
    "execute_notebook_agent",
    "get_or_run_agent",
    "run_orchestrator",
}


def load_orchestrator():
    """Carrega apenas configuração e funções do notebook, sem executar demos."""
    if not ORCHESTRATOR_NOTEBOOK.exists():
        pytest.fail(
            f"Não encontrei {ORCHESTRATOR_NOTEBOOK.name}. "
            "Coloca test_orchestrator.py na mesma pasta do notebook."
        )

    notebook = nbformat.read(ORCHESTRATOR_NOTEBOOK, as_version=4)
    namespace = {"__name__": "iach_orchestrator_under_test"}
    found = set()
    loaded_config = False

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        source = cell.source
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue  # Ignorar %pip e outras magics de notebook

        function_names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        # A célula 1 original define ROOT, OUTPUT_DIR e AGENT_CONFIG.
        # Só a executamos se também contiver AGENT_CONFIG.
        assigns_config = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "AGENT_CONFIG"
                for target in node.targets
            )
            for node in tree.body
        )

        if assigns_config or function_names.intersection(REQUIRED_FUNCTIONS):
            code = compile(source, str(ORCHESTRATOR_NOTEBOOK), "exec")
            exec(code, namespace)
            loaded_config = loaded_config or assigns_config
            found.update(function_names.intersection(REQUIRED_FUNCTIONS))

    missing = REQUIRED_FUNCTIONS - found
    if not loaded_config or missing:
        pytest.fail(
            "Não foi possível carregar a implementação do Orchestrator. "
            f"Configuração encontrada: {loaded_config}; funções em falta: {sorted(missing)}"
        )

    return namespace


@pytest.fixture
def orch(tmp_path, monkeypatch):
    """Ambiente isolado: mocks, sem executar notebooks nem tocar em dados reais."""
    ns = load_orchestrator()
    monkeypatch.setitem(ns, "OUTPUT_DIR", tmp_path / "patients")
    monkeypatch.setitem(ns, "notebook_fingerprint", lambda config: "test-version-1")
    return ns


def success_agent(agent_name, patient_id):
    return {
        "status": "success",
        "data": {"source": agent_name, "patient_id": patient_id},
    }


def test_success_and_cache(orch, monkeypatch):
    calls = []

    def fake_executor(agent_name, patient_id):
        calls.append(agent_name)
        return success_agent(agent_name, patient_id)

    monkeypatch.setitem(orch, "execute_notebook_agent", fake_executor)

    first = orch["run_orchestrator"](148611)
    assert first["status"] == "success"
    assert first["ready_for_severity"] is True
    assert len(calls) == 3

    second = orch["run_orchestrator"](148611)
    assert second["status"] == "success"
    assert len(calls) == 3, "Os agentes não deviam executar novamente com cache válida"
    assert (orch["patient_folder"](148611) / "patient_state.json").exists()


def test_one_agent_failure_blocks_severity(orch, monkeypatch):
    severity_calls = []

    def fake_executor(agent_name, patient_id):
        if agent_name == "vital":
            raise RuntimeError("Falha simulada de leitura de CSV")
        return success_agent(agent_name, patient_id)

    monkeypatch.setitem(orch, "execute_notebook_agent", fake_executor)

    result = orch["run_orchestrator"](
        148612,
        severity_runner=lambda x: severity_calls.append(x) or {"risk": 0.5},
        clean_data_loader=lambda x: {"age": 65},
    )

    assert result["status"] == "partial"
    assert result["ready_for_severity"] is False
    assert result["agents"]["vital"]["status"] == "error"
    assert not severity_calls
    assert not (orch["patient_folder"](148612) / "severity.json").exists()


def test_insufficient_data_blocks_severity(orch, monkeypatch):
    severity_calls = []

    def fake_executor(agent_name, patient_id):
        if agent_name == "laboratory":
            return {"status": "insufficient_data", "data": {}}
        return success_agent(agent_name, patient_id)

    monkeypatch.setitem(orch, "execute_notebook_agent", fake_executor)
    result = orch["run_orchestrator"](
        148613,
        severity_runner=lambda x: severity_calls.append(x) or {},
        clean_data_loader=lambda x: {"age": 65},
    )

    assert result["status"] == "partial"
    assert result["agents"]["laboratory"]["status"] == "insufficient_data"
    assert result["ready_for_severity"] is False
    assert severity_calls == []


def test_corrupted_json_reexecutes_only_that_agent(orch, monkeypatch):
    calls = []

    def fake_executor(agent_name, patient_id):
        calls.append(agent_name)
        return success_agent(agent_name, patient_id)

    monkeypatch.setitem(orch, "execute_notebook_agent", fake_executor)
    orch["run_orchestrator"](148614)
    assert len(calls) == 3

    orch["agent_output_path"](148614, "vital").write_text(
        "{invalid json", encoding="utf-8"
    )
    orch["run_orchestrator"](148614)
    assert calls.count("vital") == 2
    assert calls.count("laboratory") == 1
    assert calls.count("clinical") == 1


def test_invalid_patient_id(orch):
    for invalid_id in (-1, 0, True, "148611"):
        with pytest.raises(ValueError):
            orch["run_orchestrator"](invalid_id)


def test_all_agents_fail(orch, monkeypatch):
    def fake_executor(agent_name, patient_id):
        raise RuntimeError(f"Falha simulada: {agent_name}")

    monkeypatch.setitem(orch, "execute_notebook_agent", fake_executor)
    result = orch["run_orchestrator"](148615)
    assert result["status"] == "error"  # Requer a correção de estado global combinada
    assert result["ready_for_severity"] is False
    assert all(r["status"] == "error" for r in result["agents"].values())


def test_severity_receives_clean_features_and_agent_results(orch, monkeypatch):
    monkeypatch.setitem(orch, "execute_notebook_agent", success_agent)
    received = []

    def fake_severity(data):
        received.append(data)
        return {"status": "success", "risk": 0.3}

    result = orch["run_orchestrator"](
        148616,
        severity_runner=fake_severity,
        clean_data_loader=lambda patient_id: {"age": 65, "heart_rate": 90},
    )

    assert result["ready_for_severity"] is True
    assert len(received) == 1
    assert received[0]["patientunitstayid"] == 148616
    assert received[0]["features"]["age"] == 65
    assert set(received[0]["agent_results"]) == {"vital", "laboratory", "clinical"}
    assert (orch["patient_folder"](148616) / "severity.json").exists()


def test_missing_features_does_not_run_severity(orch, monkeypatch):
    monkeypatch.setitem(orch, "execute_notebook_agent", success_agent)
    severity_calls = []

    with pytest.raises(ValueError, match="Missing or invalid cleaned patient features"):
        orch["run_orchestrator"](
            148617,
            severity_runner=lambda data: severity_calls.append(data) or {},
            clean_data_loader=lambda patient_id: {},
        )

    assert severity_calls == []
