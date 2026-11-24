"""IACH terminal entry: patient ID -> Orchestrator -> overview -> human review.

Default: a concise progress display; detailed Orchestrator output is saved to a log.
Use --verbose to see the original detailed messages during execution.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path


def _patient_id(raw: str) -> int:
    try:
        value = int(raw.strip())
    except (ValueError, AttributeError) as exc:
        raise ValueError("Patient ID must be a positive integer.") from exc
    if value <= 0:
        raise ValueError("Patient ID must be a positive integer.")
    return value


def _status(entry: object) -> str:
    return str(entry.get("status", "unknown")) if isinstance(entry, dict) else str(entry)


def _print_pipeline_summary(state: dict) -> None:
    agents = state.get("agents") or {}
    if isinstance(agents, dict):
        statuses = {name: _status(agents.get(name, "not_run"))
                    for name in ("vital", "laboratory", "clinical")}
        if all(status == "success" for status in statuses.values()):
            print("[OK] Patient data collected (Vital, Laboratory, Clinical)")
        else:
            print("[INFO] Data agents:", ", ".join(f"{name}={status}"
                                                         for name, status in statuses.items()))

    for key, label in (("severity", "Severity assessment"),
                       ("explanation", "Explanation"),
                       ("critic", "Critic review")):
        result = state.get(key)
        status = _status(result)
        if key == "critic" and status == "approved":
            print("[OK] Critic review approved")
        elif status == "success":
            print(f"[OK] {label} completed")
        elif status not in ("not_run", "None", "unknown"):
            print(f"[INFO] {label}: {status}")


def _log_path(folder: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return folder / "logs" / f"pipeline_{stamp}.log"


def launch(patient_id: int, *, resume: bool = False, rerun: bool = False,
           reviewer: str = "researcher", verbose: bool = False) -> bool:
    from orchestration import orchestrator as orch
    from scripts.researcher_review import run_review_cli

    folder = Path(orch.OUTPUT_DIR) / str(patient_id)
    previous = (folder / "patient_state.json").is_file()
    if previous and not resume and not rerun:
        answer = input("A saved analysis exists. [Enter] resume / [n] new run: ").strip().lower()
        if answer in ("", "r", "resume"):
            resume = True
        elif answer in ("n", "new"):
            rerun = True
        else:
            print("[CANCELLED] Choose Enter to resume or n to start a new run.")
            return False

    if resume:
        if not previous:
            print("[ERROR] No saved analysis for this patient. Start a new analysis instead.")
            return False
        print(f"[RESUME] Reopening saved review for patient {patient_id}; no IAedu calls.")
    else:
        print(f"\n=== IACH | Analyzing patient {patient_id} ===")
        print("[RUN] Processing patient data, model prediction and explanation...")
        log_file = None
        try:
            if verbose:
                state = orch.run_orchestrator(patient_id)
            else:
                log_file = _log_path(folder)
                log_file.parent.mkdir(parents=True, exist_ok=True)
                # Scope the redirection to the pipeline only. The interactive
                # review must continue to use the normal terminal streams.
                with log_file.open("w", encoding="utf-8") as stream:
                    with redirect_stdout(stream), redirect_stderr(stream):
                        state = orch.run_orchestrator(patient_id)
        except Exception as exc:
            # Never print exception messages: API errors could contain secrets.
            print(f"[ERROR] Pipeline could not complete ({type(exc).__name__}).")
            if log_file is not None:
                print(f"[INFO] Pipeline log: {log_file}")
            return False
        if not isinstance(state, dict):
            print("[ERROR] Orchestrator returned an unexpected result.")
            return False
        _print_pipeline_summary(state)
        if not (isinstance(state.get("explanation"), dict) and
                isinstance(state.get("critic"), dict)):
            print(f"[STOP] Report unavailable (pipeline status: {state.get('status')}).")
            if log_file is not None:
                print(f"[INFO] Pipeline log: {log_file}")
            print("Review could not be opened; inspect the pipeline log for details.")
            return False
        print("[OK] Analysis finished. Opening Researcher Review...\n")
        if verbose:
            print("[INFO] Verbose pipeline output was displayed above.")

    run_review_cli(patient_id, reviewer=reviewer, show_overview=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="IACH: analyze a patient and open researcher review")
    parser.add_argument("patient_id", nargs="?", type=int, help="eICU patientunitstayid")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Reuse saved results without rerunning agents")
    mode.add_argument("--rerun", action="store_true", help="Start a new pipeline run, preserving older history")
    parser.add_argument("--verbose", action="store_true", help="Show detailed Orchestrator messages")
    parser.add_argument("--reviewer", default="researcher")
    args = parser.parse_args()
    try:
        patient_id = _patient_id(str(args.patient_id) if args.patient_id is not None else
                                 input("Patient ID: "))
    except (ValueError, EOFError, KeyboardInterrupt) as exc:
        parser.exit(2, f"[ERROR] {exc if isinstance(exc, ValueError) else 'No patient ID provided.'}\n")
    if not launch(patient_id, resume=args.resume, rerun=args.rerun,
                  reviewer=args.reviewer, verbose=args.verbose):
        parser.exit(1)


if __name__ == "__main__":
    main()
