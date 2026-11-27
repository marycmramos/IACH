"""IACH — unified terminal interface for analysis and researcher review.
From project root:
  python -m scripts.start_iach
  python -m scripts.start_iach 176730 --resume
  python -m scripts.start_iach 176730 --rerun
  python -m scripts.start_iach 176730 --verbose
One file for patient selection, report viewing, feedback and human approval.
The existing orchestration/human_review backend modules remain unchanged.
"""
from __future__ import annotations
import argparse
import json
import re
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from orchestration.human_review import (
    HumanReviewError, approve_review, current_version, get_review, request_revision,
)
FEATURE_LABELS = {
    "rr_last": "last recorded respiratory rate",
    "rr_max": "maximum recorded respiratory rate",
    "rr_mean": "average respiratory rate",
    "age": "age",
    "sbp_last": "last systolic blood pressure",
    "sbp_mean": "average systolic blood pressure",
    "sbp_max": "maximum systolic blood pressure",
    "dbp_max": "maximum diastolic blood pressure",
    "map_mean": "average mean arterial pressure",
    "hr_last": "last recorded heart rate",
    "hr_mean": "average heart rate",
    "BUN_max": "maximum BUN (blood urea nitrogen)",
    "BUN_last": "last BUN (blood urea nitrogen)",
}
def show_critic_issues(critic: dict) -> None:
    """Display actionable Critic feedback for the investigator."""
    issues = critic.get("issues") or []
    if not issues:
        if critic.get("status") == "approved":
            print("[CRITIC AGENT] Approved — No issues identified.")
        elif critic.get("status") == "review_error":
            print("[CRITIC AGENT] Review incomplete; this is NOT an approval.")
        return
    print(f"\n[CRITIC AGENT] Review required — identified {len(issues)} issue(s):")
    for number, issue in enumerate(issues, start=1):
        print(f"\n{number}. {issue.get('code', 'unknown')} [{issue.get('severity', '?')}]")
        for label, key in (("Claim", "claim"), ("Reason", "reason"),
                           ("Evidence", "evidence"), ("Suggestion", "suggestion")):
            value = issue.get(key)
            if value:
                print(f"   {label}: {value}")
def show_report(version: dict) -> None:
    report = version["explanation"]["explanation"]
    for name in ("mortality_summary", "clinical_summary", "shap_summary", "overall_explanation"):
        print(f"\n--- {name.replace('_', ' ').title()} ---\n{report.get(name, '(missing)')}")
    print("\n--- Limitations ---")
    for note in report.get("limitations", []):
        print(f"- {note}")
    critic = version["critic"]
    print(f"\nCritic Agent status: {critic['status']}")
    show_critic_issues(critic)
def readable_text(text: str) -> str:
    """Presentation-only substitutions; never change source JSON values."""
    value = str(text)
    for code, description in FEATURE_LABELS.items():
        value = re.sub(r"\b" + re.escape(code) + r"\b", description, value)
    # Round unnecessarily precise decimal numbers for narrative display only.
    return re.sub(r"(?<![\w.])(-?\d+\.\d{6,})(?![\w.])",
                  lambda match: f"{float(match.group(1)):.3f}", value)
def show_feedback_compliance(critic: dict) -> None:
    check = critic.get("feedback_compliance")
    if not isinstance(check, dict):
        return  # Original automatic run or older review version.
    status = check.get("status", "review_error")
    print(f"\n[CRITIC AGENT — HUMAN FEEDBACK] {status}")
    if check.get("summary"):
        print(f"  Summary: {check['summary']}")
    for index, requirement in enumerate(check.get("requirements") or [], 1):
        print(f"  {index}. {requirement.get('requirement', 'Requirement')} "
              f"[{requirement.get('status', '?')}]")
        print(f"     Evidence: {requirement.get('evidence', 'Not available')}")
        print(f"     Suggestion: {requirement.get('suggestion', 'Not available')}")
    if status != 'satisfied':
        print("[INFO] This is an assessment of the researcher's request, "
              "not a factual or clinical verdict. The investigator decides.")
def _format_readable_number(value: object, decimals: int = 1) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "not available"
    return f"{value:.{decimals}f}"
def _feature_name(code: str) -> str:
    return FEATURE_LABELS.get(code, code.replace("_", " "))
def show_accessible_report(version: dict) -> None:
    """Short plain-language view, derived from stored evidence (no model call)."""
    result = version.get("explanation") or {}
    report = result.get("explanation") or {}
    prediction = result.get("mortality_prediction") or {}
    probability = prediction.get("probability")
    print("\n--- At a Glance ---")
    if isinstance(probability, (int, float)) and not isinstance(probability, bool) and 0 <= probability <= 1:
        print(f"The model estimated an ICU mortality probability of {probability * 100:.2f}% "
              "for this record. This is a model estimate, not a clinical prognosis.")
    else:
        print(readable_text(report.get("mortality_summary", "Model estimate unavailable.")))
    apache_versions = (result.get("apache_comparison") or {}).get("versions") or {}
    comparisons = []
    for name, entry in apache_versions.items():
        if not isinstance(entry, dict):
            continue
        reference_probability = entry.get("apache_probability")
        if isinstance(reference_probability, (int, float)) and not isinstance(reference_probability, bool) and 0 <= reference_probability <= 1:
            comparisons.append(f"APACHE {name}: {reference_probability * 100:.2f}%")
    if comparisons:
        print("For context only, separate APACHE estimates were " + "; ".join(comparisons) +
              ". These estimates are not directly interchangeable with the XGBoost estimate.")
    clinical = report.get("clinical_summary") or ""
    if clinical:
        # Keep only the first sourced sentence rather than repeating the full clinical narrative.
        context = clinical.split(". ", 1)[0].strip()
        if context and not context.endswith("."):
            context += "."
        print("\n--- Clinical Context ---")
        print(readable_text(context))
    print("\n--- What Influenced the Model? ---")
    features = (result.get("shap_explanation") or {}).get("top_features") or []
    selected = [f for f in features if isinstance(f, dict) and f.get("direction") in ("increases", "decreases")]
    if selected:
        print("SHAP helps show which recorded or estimated inputs moved the model's "
              "prediction up or down compared with its reference. "
              "It does not tell us what caused the patient's outcome.")
        for direction, heading in (("increases", "Increased the model prediction"),
                                   ("decreases", "Decreased the model prediction")):
            matches = [f for f in selected if f.get("direction") == direction][:2]
            if matches:
                print(f"{heading}:")
            for feature in matches:
                code = str(feature.get("feature", "unknown"))
                original = feature.get("original_value")
                if feature.get("was_imputed"):
                    detail = " (model-filled value; not a measurement)"
                elif isinstance(original, (int, float)) and not isinstance(original, bool):
                    detail = f" (recorded value: {_format_readable_number(original)})"
                else:
                    detail = ""
                print(f"  - {_feature_name(code)}{detail}")
        print("The exact SHAP values and all variables are available in 'View technical details'.")
    else:
        print(readable_text(report.get("shap_summary", "SHAP information unavailable.")))
    print("\n--- Important Limitations ---")
    notes = report.get("limitations") or []
    if notes:
        for note in notes:
            print(f"- {readable_text(note)}")
    else:
        print("No limitations supplied with this report; check the full report before approval.")
    critic = version.get("critic") or {}
    print(f"\nCritic Agent factual status: {critic.get('status', 'unknown')}")
    show_critic_issues(critic)
    show_feedback_compliance(critic)
def show_technical_report(version: dict) -> None:
    """Structured model inputs and outputs only; do not repeat the narrative."""
    result = version.get("explanation") or {}
    prediction = result.get("mortality_prediction") or {}
    print("\n--- Model Prediction (raw values) ---")
    for key in ("model", "outcome", "probability"):
        print(f"{key}: {prediction.get(key, 'not available')}")
    apache = result.get("apache_comparison") or {}
    print("\n--- APACHE Comparison (descriptive only) ---")
    versions = apache.get("versions") or {}
    if not versions:
        print("No APACHE comparison supplied.")
    for name, entry in versions.items():
        if isinstance(entry, dict):
            print(f"APACHE {name}: score={entry.get('apache_score')}, "
                  f"probability={entry.get('apache_probability')}, "
                  f"difference_percentage_points={entry.get('difference_percentage_points')}")
    if apache.get("note"):
        print(f"Note: {apache['note']}")
    shap = result.get("shap_explanation") or {}
    print("\n--- SHAP Feature Values (full precision) ---")
    print(f"Method: {shap.get('method', 'unknown')} | Unit: {shap.get('unit', 'unknown')} "
          f"| Base probability: {shap.get('base_probability', 'not available')}")
    features = shap.get("all_features") or shap.get("top_features") or []
    if not features:
        print("No SHAP features supplied.")
    for feature in features:
        if isinstance(feature, dict):
            print(f"- {feature.get('feature')}: SHAP={feature.get('shap_value')}, "
                  f"original={feature.get('original_value')}, "
                  f"processed={feature.get('processed_value')}, "
                  f"imputed={feature.get('was_imputed')}, "
                  f"direction={feature.get('direction')}")
    critic = version.get("critic") or {}
    print(f"\nCritic Agent factual status: {critic.get('status', 'unknown')}")
    show_critic_issues(critic)
    show_feedback_compliance(critic)
def show_initial_overview(session: dict) -> None:
    """Display a concise overview of the current patient report."""

    version = current_version(session)
    result = version.get("explanation") or {}
    report = result.get("explanation") or {}
    critic = version.get("critic") or {}
    prediction = result.get("mortality_prediction") or {}

    print("\n" + "=" * 60)
    print(f"IACH | Patient {session['patientunitstayid']}")
    print(
        f"Version {version['version']} | "
        f"Critic: {critic.get('status', 'unknown')} | "
        f"Human: {session.get('status', 'unknown')}"
    )
    print("=" * 60)

    # Mortality prediction
    probability = prediction.get("probability")

    print("\nESTIMATED ICU MORTALITY")

    if (
        isinstance(probability, (int, float))
        and not isinstance(probability, bool)
        and 0 <= probability <= 1
    ):
        print(f"{probability * 100:.2f}%")
    else:
        print("Unavailable")

    print("Research model estimate — not clinically validated.")

    # Clinical context
    print("\nClinical Context")

    clinical = report.get("clinical_summary") or ""

    if clinical:
        context = clinical.split(". ", 1)[0].strip()
        print(readable_text(context.rstrip(".") + "."))
    else:
        print("Clinical context unavailable.")

    # SHAP contributions
    print("\nWhat Influenced the Model?")

    features = (
        result.get("shap_explanation") or {}
    ).get("top_features") or []

    for direction, label in (
        ("increases", "Higher model output"),
        ("decreases", "Lower model output")
    ):
        selected = [
            feature for feature in features
            if isinstance(feature, dict)
            and feature.get("direction") == direction
        ][:2]

        for feature in selected:
            name = _feature_name(
                str(feature.get("feature", "unknown"))
            )

            marker = (
                " [imputed; not measured]"
                if feature.get("was_imputed")
                else ""
            )

            print(f"  - {label}: {name}{marker}")

    if not any(
        isinstance(feature, dict)
        and feature.get("direction") in ("increases", "decreases")
        for feature in features
    ):
        print("SHAP contributions unavailable.")

    print("SHAP explains model behavior, not clinical causality.")

    # Limitations
    print("\nImportant Limitations")

    limitations = report.get("limitations") or []

    if limitations:
        for note in limitations[:2]:
            print(f"  - {readable_text(note)}")

        if len(limitations) > 2:
            print(
                f"  + {len(limitations) - 2} additional limitations"
                " available in option 6."
            )
    else:
        print("No limitations supplied; consult the full report.")

    # Review status
    feedback = critic.get("feedback_compliance")

    if isinstance(feedback, dict):
        print(
            f"\nFeedback evaluation: "
            f"{feedback.get('status', 'review_error')}"
        )

    if (
        critic.get("status") != "approved"
        or critic.get("approved") is not True
    ):
        print("\n[WARNING] Critic approval required.")
    elif session.get("status") != "approved":
        print("\nHuman approval pending.")

    print("\n" + "=" * 60)


def show_history(session: dict) -> None:
    print(f"\nAutomatic run: {session['run_id']}")
    for version in session["versions"]:
        print(f"  Version {version['version']}: {version['source']}, Critic={version['critic']['status']}")
    for decision in session["decisions"]:
        extra = f", result={decision.get('result') }" if decision['type'] == 'revision_requested' else ""
        print(f"  Decision: {decision['type']} on version {decision['version']}{extra} ({decision['created_at']})")

def run_review_cli(
    patient_id: int,
    reviewer: str = "researcher",
    show_overview: bool = True
) -> None:

    try:
        session = get_review(patient_id)
    except HumanReviewError as exc:
        print(f"[ERROR] {exc}")
        return

    if show_overview:
        show_initial_overview(session)

    while True:
        version = current_version(session)
        critic = version["critic"]

        print("\nActions")
        print("1. Approve report")
        print("2. Request revision")
        print("3. View full report")
        print("4. View history")
        print("5. View technical details")
        print("6. View accessible explanation")
        print("7. Exit")

        try:
            choice = input("Option: ").strip()
            if choice == "1":
                if session["status"] == "approved":
                    print("[INFO] This session is already approved.")
                    continue
                if critic.get("status") != "approved" or critic.get("approved") is not True:
                    print("[INFO] Cannot approve: Critic Agent has not approved this version.")
                    show_critic_issues(critic)
                    continue
                check = critic.get("feedback_compliance") or {}
                if check and check.get("status") != "satisfied":
                    print(f"[NOTICE] Human feedback evaluation: {check.get('status')}. "
                          "Please read the unmet requirements before approving.")
                    show_feedback_compliance(critic)
                print(f"You are about to approve version {version['version']} for patient {patient_id}.")
                print("This is approval of a research report, NOT clinical validation.")
                confirmation = input("Confirm researcher approval? [y/N]: ").strip().lower()
                if confirmation not in {"y", "yes"}:
                    print("[CANCELLED] No approval was recorded.")
                    continue
                session = approve_review(patient_id, reviewer)
                print(f"[OK] Investigator approved version {session['approved_version']}.")
                print("This is human approval of a research report, not clinical validation.")
                print("[INFO] Returning to patient selection.")
                return
            elif choice == "2":
                print("Write your feedback (press Enter on a blank line to submit):")
                lines = []
                while True:
                    line = input()
                    if not line.strip():
                        break
                    lines.append(line)
                feedback = "\n".join(lines)
                print("[RUN] Explanation revision and real Critic review...")
                session = request_revision(patient_id, feedback, reviewer)
                latest_critic = current_version(session)["critic"]
                print(f"[OK] New version {session['current_version']}, Critic Agent: {latest_critic['status']}")
                print(f"\n[INFO] Revised version {session['current_version']} is ready for review.")
                show_initial_overview(session)
                print("[INFO] Select option 6 for the complete accessible explanation.")
                if latest_critic.get("status") == "approved":
                    print("[INFO] Awaiting explicit researcher approval (option 1).")
                else:
                    print("[INFO] Report is not approved by the Critic Agent; review the issues before deciding.")
            elif choice == "3":
                show_report(version)
                show_feedback_compliance(critic)
            elif choice == "4":
                show_history(session)
            elif choice == "5":
                show_technical_report(version)
            elif choice == "6":
                show_accessible_report(version)
            elif choice == "7":
                print("[EXIT] Session saved. You can resume with the same command.")
                return
            else:
                print("Invalid option; enter 1–7.")
        except HumanReviewError as exc:
            print(f"[ERROR] {exc}")
            session = get_review(patient_id)
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT] Session saved.")
            return
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
def _patient_selection_loop(*, reviewer: str, verbose: bool) -> None:
    """Interactive entry point; return here after a review is approved or exited."""
    while True:
        try:
            raw = input("\nPatient ID (or q to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT] IACH session ended.")
            return
        if raw.lower() in {"q", "quit", "exit"}:
            print("[EXIT] IACH session ended.")
            return
        try:
            patient_id = _patient_id(raw)
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            continue
        try:
            launch(patient_id, reviewer=reviewer, verbose=verbose)
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT] IACH session ended.")
            return
def main() -> None:
    parser = argparse.ArgumentParser(description="IACH: analyze a patient and open researcher review")
    parser.add_argument("patient_id", nargs="?", type=int, help="eICU patientunitstayid")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Reuse saved results without rerunning agents")
    mode.add_argument("--rerun", action="store_true", help="Start a new pipeline run, preserving older history")
    parser.add_argument("--verbose", action="store_true", help="Show detailed Orchestrator messages")
    parser.add_argument("--reviewer", default="researcher")
    args = parser.parse_args()
    if args.patient_id is None:
        if args.resume or args.rerun:
            parser.error("--resume and --rerun require an explicit patient ID")
        _patient_selection_loop(reviewer=args.reviewer, verbose=args.verbose)
        return
    try:
        patient_id = _patient_id(str(args.patient_id))
    except ValueError as exc:
        parser.exit(2, f"[ERROR] {exc}\n")
    if not launch(patient_id, resume=args.resume, rerun=args.rerun,
                  reviewer=args.reviewer, verbose=args.verbose):
        parser.exit(1)
if __name__ == "__main__":
    main()
