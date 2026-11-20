"""IACH orchestration: run the three Data Agents, cache results, optionally run severity.

Run from the project root:
    python -m orchestration.orchestrator 148611
    python -m orchestration.orchestrator 148611 --with-severity

This is an academic prototype, not a clinically validated decision tool.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "patients"
DATA_VERSION = "eicu-clean-v1"

# Preserve the order used in the original notebook.
AGENT_CONFIG = {
    "vital": {
        "module": "agents.vital.vital_agent",
        "function": "run_vital_agent",
        "sources": ["processed_data/vital_agent_data.csv"],
    },
    "laboratory": {
        "module": "agents.laboratory.laboratory_agent",
        "function": "run_lab_agent",
        "sources": ["processed_data/lab_agent_data.csv", "processed_data/lab_agent_raw.csv"],
    },
    "clinical": {
        "module": "agents.clinical.clinical_agent",
        "function": "run_clinical_agent",
        "sources": ["processed_data/clinical_agent_data.csv"],
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def patient_folder(patient_id: int) -> Path:
    folder = OUTPUT_DIR / str(patient_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def agent_output_path(patient_id: int, agent_name: str) -> Path:
    return patient_folder(patient_id) / f"{agent_name}.json"


def agent_fingerprint(config: dict) -> str:
    """Version agent code, input CSVs and configured data version.

    Uses file contents for the module and size/mtime for large datasets.
    Existing notebook-based cache files will naturally be invalidated.
    """
    module_file = ROOT.joinpath(*config["module"].split(".")).with_suffix(".py")
    if not module_file.is_file():
        raise FileNotFoundError(f"Agent module file missing: {module_file}")
    digest = hashlib.sha256()
    digest.update(DATA_VERSION.encode("utf-8"))
    digest.update(config["module"].encode("utf-8"))
    digest.update(module_file.read_bytes())
    for relative in config["sources"]:
        source = ROOT / relative
        stat = source.stat()  # ensure required source exists
        digest.update(f"{relative}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()


def to_jsonable(value):
    """Convert Pydantic, numpy-style scalars and nested structures into JSON-safe values."""
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


def save_json(path: Path, data: dict) -> None:
    """Write JSON via a temporary file to avoid partial results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2,
                                  allow_nan=False), encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load_cached_agent(patient_id: int, agent_name: str) -> dict | None:
    path = agent_output_path(patient_id, agent_name)
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (
            isinstance(cached, dict)
            and cached.get("patientunitstayid") == patient_id
            and cached.get("agent") == agent_name
            and cached.get("status") == "success"
            and cached.get("fingerprint") == agent_fingerprint(AGENT_CONFIG[agent_name])
            and isinstance(cached.get("data"), dict)
            and bool(cached["data"])
            and cached["data"].get("status") == "success"
            and isinstance(cached["data"].get("agent_output"), dict)
            and cached["data"]["agent_output"].get("status") == "success"
        ):
            return cached
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def execute_agent(agent_name: str, patient_id: int) -> dict:
    config = AGENT_CONFIG[agent_name]
    module = importlib.import_module(config["module"])
    runner = getattr(module, config["function"])
    return to_jsonable(runner(patient_id))


def get_or_run_agent(agent_name: str, patient_id: int) -> dict:
    if agent_name not in AGENT_CONFIG:
        raise ValueError(f"Unknown Data Agent: {agent_name}")
    cached = load_cached_agent(patient_id, agent_name)
    if cached is not None:
        print(f"[CACHE] {agent_name}")
        return cached

    print(f"[RUN] {agent_name}")
    fingerprint = agent_fingerprint(AGENT_CONFIG[agent_name])
    try:
        data = execute_agent(agent_name, patient_id)
        if not isinstance(data, dict):
            raise TypeError("Data Agent must return a structured dictionary or Pydantic model")
        status = data.get("status")
        if status not in {"success", "insufficient_data", "error"}:
            raise ValueError(f"Invalid Data Agent status: {status}")
        if data.get("patient_id") != patient_id:
            raise ValueError("Data Agent returned a different patient ID")
        agent_output = data.get("agent_output")
        if not isinstance(agent_output, dict) or agent_output.get("status") != status:
            raise ValueError("Data Agent state and agent_output status disagree")
        if agent_output.get("patient_id") != patient_id:
            raise ValueError("Data Agent output returned a different patient ID")
        error = data.get("error_message")
    except Exception as exc:
        status, data = "error", None
        error = f"{type(exc).__name__}: {str(exc)[:350]}"
        print(f"[ERROR] {agent_name}: {error}")

    result = {
        "patientunitstayid": patient_id,
        "agent": agent_name,
        "status": status,
        "data": data,
        "error": error,
        "fingerprint": fingerprint,
        "created_at": now(),
    }
    save_json(agent_output_path(patient_id, agent_name), result)
    return result


def load_ml_features(patient_id: int) -> dict:
    """Get exactly the model's feature columns for one eICU stay."""
    import pandas as pd
    from ml_model.predict import load_model

    path = ROOT / "ml_model" / "ml_features_24h.csv"
    _, feature_names = load_model()
    df = pd.read_csv(path)
    matching = df.loc[df["patientunitstayid"] == patient_id]
    if len(matching) != 1:
        raise ValueError(f"Expected one ML row for {patient_id}, found {len(matching)}")
    row = matching.iloc[0]
    missing_columns = [name for name in feature_names if name not in df.columns]
    if missing_columns:
        raise ValueError(f"ML features missing columns: {missing_columns}")
    return {
        name: None if pd.isna(row[name]) else to_jsonable(row[name])
        for name in feature_names
    }


def run_orchestrator(patient_id: int) -> dict:
    """Run the three Data Agents and automatically assess severity.

    Severity runs only when all three Data Agents succeed. APACHE IV/IVa
    results are optional, but ML features and the trained model are required.
    """
    if isinstance(patient_id, bool) or not isinstance(patient_id, int) or patient_id <= 0:
        raise ValueError("patient_id must be a positive integer")

    print(f"\n=== IACH | Patient {patient_id} ===")
    agents = {name: get_or_run_agent(name, patient_id) for name in AGENT_CONFIG}
    statuses = [result["status"] for result in agents.values()]
    all_success = all(status == "success" for status in statuses)
    overall_status = (
        "success" if all_success else
        "error" if all(status == "error" for status in statuses) else "partial"
    )
    state = {
        "patientunitstayid": patient_id,
        "status": overall_status,
        "agents": agents,
        "ready_for_severity": all_success,
        "created_at": now(),
    }
    state_path = patient_folder(patient_id) / "patient_state.json"
    save_json(state_path, state)

    if not all_success:
        print("[STOP] Data Agents incomplete; severity not run")
        return state

    try:
        from agents.severity.severity_agent import run_severity_agent
        from agents.severity.apache_loader import load_apache

        features = load_ml_features(patient_id)
        if not isinstance(features, dict) or not features:
            raise ValueError("Missing or invalid ML features")

        # No APACHE record or no local source file should not block XGBoost.
        # Other failures are surfaced rather than silently treated as missing data.
        try:
            apache_result = load_apache(patient_id)
        except FileNotFoundError:
            apache_result = None
            print("[WARN] APACHE file unavailable; proceeding without comparison")

        severity_input = {
            "patientunitstayid": patient_id,
            "features": features,
            "agent_results": {name: result["data"] for name, result in agents.items()},
            "apache_result": apache_result,
        }
        print("[RUN] Severity Agent")
        severity = to_jsonable(run_severity_agent(severity_input))
        if not isinstance(severity, dict) or severity.get("status") != "success":
            raise ValueError("Severity Agent did not complete successfully")
        if str(severity.get("patientunitstayid")) != str(patient_id):
            raise ValueError("Severity Agent returned a different patient ID")
        state["severity"] = severity
        save_json(patient_folder(patient_id) / "severity.json", {
            "patientunitstayid": patient_id,
            "result": severity,
            "created_at": now(),
        })
        print("[OK] Severity assessment completed")

        # Run Explanation Agent after Severity
        try:
            from agents.explanation.explanation_agent import run_explanation_agent

            explanation_input = {
                "patientunitstayid": patient_id,
                "severity_result": severity,
                "agent_results": {
                    name: result["data"]
                    for name, result in agents.items()
                }
            }

            print("[RUN] Explanation Agent")

            explanation = to_jsonable(
                run_explanation_agent(
                    explanation_input,
                    use_llm=True
                )
            )

            if explanation.get("status") != "success":
                raise ValueError(
                    f"Explanation Agent failed: {explanation.get('error')}"
                )

            state["explanation"] = explanation

            save_json(
                patient_folder(patient_id) / "explanation.json",
                {
                    "patientunitstayid": patient_id,
                    "result": explanation,
                    "created_at": now()
                }
            )

            print("[OK] Explanation Agent completed")

        except Exception as exc:
            state["status"] = "partial"
            state["explanation_error"] = (
                f"{type(exc).__name__}: {str(exc)[:350]}"
            )
            print(f"[ERROR] Explanation Agent: {state['explanation_error']}")
    except Exception as exc:
        state["status"] = "partial"
        state["severity_error"] = f"{type(exc).__name__}: {str(exc)[:350]}"
        print(f"[ERROR] Severity Agent: {state['severity_error']}")
    finally:
        save_json(state_path, state)
    return state


def run_full_pipeline(patient_id: int) -> dict:
    """Backward-compatible entry point; all agents now run by default."""
    return run_orchestrator(patient_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="IACH research orchestrator")
    parser.add_argument("patient_id", type=int, help="eICU patientunitstayid")
    args = parser.parse_args()
    result = run_orchestrator(args.patient_id)
    print(json.dumps({
        "patientunitstayid": result["patientunitstayid"],
        "status": result["status"],
        "ready_for_severity": result["ready_for_severity"],
        "agents": {name: entry["status"] for name, entry in result["agents"].items()},
        "severity": result.get("severity", {}).get("status", "not_run"),
        "severity_error": result.get("severity_error"),
        "explanation": result.get("explanation", {}).get("status", "not_run"),
        "explanation_error": result.get("explanation_error"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
