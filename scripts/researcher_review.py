"""IACH investigator review CLI. From repo root:

    python -m scripts.researcher_review 156906
"""
from __future__ import annotations

import argparse
import json
import re

from orchestration.human_review import (
    HumanReviewError, approve_review, current_version, get_review, request_revision,
)


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


def show_history(session: dict) -> None:
    print(f"\nAutomatic run: {session['run_id']}")
    for version in session["versions"]:
        print(f"  Version {version['version']}: {version['source']}, Critic={version['critic']['status']}")
    for decision in session["decisions"]:
        extra = f", result={decision.get('result') }" if decision['type'] == 'revision_requested' else ""
        print(f"  Decision: {decision['type']} on version {decision['version']}{extra} ({decision['created_at']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="IACH investigator review (terminal)")
    parser.add_argument("patient_id", type=int, help="eICU patientunitstayid")
    parser.add_argument("--reviewer", default="researcher", help="Name or identifier of the reviewer")
    args = parser.parse_args()
    try:
        session = get_review(args.patient_id)
    except HumanReviewError as exc:
        parser.exit(1, f"[ERROR] {exc}\nRun the regular Orchestrator for this patient first.\n")

    print("\n=== IACH | Researcher Review ===")
    while True:
        version = current_version(session)
        critic = version["critic"]
        feedback_status = (critic.get("feedback_compliance") or {}).get("status")
        feedback_label = f" | Feedback: {feedback_status}" if feedback_status else ""
        print(f"\nPatient: {args.patient_id} | Version: {version['version']} | Critic: {critic['status']} | Human: {session['status']}{feedback_label}")
        print("1. Approve report\n2. Request revision\n3. View full report\n4. View history\n5. View technical details\n6. View accessible explanation\n7. Exit")
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
                print(f"You are about to approve version {version['version']} for patient {args.patient_id}.")
                print("This is approval of a research report, NOT clinical validation.")
                confirmation = input("Confirm researcher approval? [y/N]: ").strip().lower()
                if confirmation not in {"y", "yes"}:
                    print("[CANCELLED] No approval was recorded.")
                    continue
                session = approve_review(args.patient_id, args.reviewer)
                print(f"[OK] Investigator approved version {session['approved_version']}.")
                print("This is human approval of a research report, not clinical validation.")
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
                session = request_revision(args.patient_id, feedback, args.reviewer)
                latest_critic = current_version(session)["critic"]
                print(f"[OK] New version {session['current_version']}, Critic Agent: {latest_critic['status']}")
                print(f"\n========== REVISED EXPLANATION — VERSION {session['current_version']} ==========")
                revised = current_version(session)
                show_accessible_report(revised)
                print("\n====================================================")
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
            session = get_review(args.patient_id)
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT] Session saved.")
            return


if __name__ == "__main__":
    main()
