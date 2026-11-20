"""Controlled LIVE test of IACH's Critic->Explanation revision loop.

Run ONLY on a local configured IACH repository:
    python scripts/live_feedback_loop_test.py --confirm-live-test

The first report is deliberately wrong and clearly marked SYNTHETIC. Data Agent
cache is copied to an isolated test output folder. The *real* Severity Agent,
Critic Agent and IAedu-backed Explanation revision are used thereafter.
Normal patient outputs and source code are not changed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid


def make_incorrect_initial_report(real_template: dict) -> dict:
    """Alter only explanatory prose, never underlying clinical/model evidence."""
    result = deepcopy(real_template)
    if result.get("status") != "success" or not isinstance(result.get("explanation"), dict):
        raise ValueError("A successful structured template explanation is required")
    result["generation_method"] = "synthetic_test_injection"
    report = result["explanation"]
    report["mortality_summary"] = "The XGBoost model mortality risk is 25.4%."
    report["clinical_summary"] = "The measured BUN_max was 21.0."
    report["shap_summary"] = "The elevated respiratory rate caused a higher mortality risk."
    report["overall_explanation"] = (
        "The XGBoost model mortality risk is 25.4%. The measured BUN_max was 21.0. "
        "The elevated respiratory rate caused a higher mortality risk. "
        "The XGBoost model is clinically superior to APACHE IV."
    )
    return result


def run_live_test(patient_id: int = 156906, max_revisions: int = 2) -> dict:
    import orchestration.orchestrator as orch
    import agents.explanation.explanation_agent as explanation_module

    root = Path(orch.ROOT)
    normal_output_dir = Path(orch.OUTPUT_DIR)
    test_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    isolated_output_dir = root / "outputs" / "feedback_loop_tests" / test_id / "patients"
    target_dir = isolated_output_dir / str(patient_id)
    target_dir.mkdir(parents=True, exist_ok=False)

    # Warm cache only: a changed fingerprint is rejected and rerun by orchestrator.
    source_dir = normal_output_dir / str(patient_id)
    for agent_name in ("vital", "laboratory", "clinical"):
        cached = source_dir / f"{agent_name}.json"
        if cached.is_file():
            shutil.copy2(cached, target_dir / cached.name)

    original_agent = explanation_module.run_explanation_agent
    calls = {"injections": 0, "revision_calls": 0}

    def controlled_agent(patient_state, *args, **kwargs):
        if kwargs.get("revision_context") is not None:
            calls["revision_calls"] += 1
            # This is the REAL IAedu call, without injecting revised text.
            return original_agent(patient_state, *args, **kwargs)
        if calls["injections"] != 0:
            raise RuntimeError("The synthetic first-round injection was requested twice")
        calls["injections"] += 1
        # Only the first explanation is synthetic; use the real agent's template
        # contract to preserve all structured non-narrative outputs.
        template = original_agent(patient_state, use_llm=False)
        return make_incorrect_initial_report(template)

    print("[TEST ONLY] First explanation is intentionally incorrect and synthetic.")
    print("[TEST ONLY] Critic and subsequent IAedu revisions are REAL.")
    print(f"[TEST ONLY] Isolated outputs: {isolated_output_dir}")
    try:
        orch.OUTPUT_DIR = isolated_output_dir
        explanation_module.run_explanation_agent = controlled_agent
        state = orch.run_orchestrator(patient_id, max_revisions=max_revisions)
    finally:
        explanation_module.run_explanation_agent = original_agent
        orch.OUTPUT_DIR = normal_output_dir

    review = state.get("review_cycle") or {}
    history = review.get("history") or []
    summary = {
        "test_mode": "controlled_synthetic_first_explanation",
        "patientunitstayid": patient_id,
        "overall_status": state.get("status"),
        "first_round_rejected": bool(history and history[0].get("critic_status") == "needs_revision"),
        "real_revision_called": calls["revision_calls"] > 0,
        "latest_critic_status": (state.get("critic") or {}).get("status"),
        "history": history,
        "stop_reason": review.get("stop_reason"),
        "test_output_folder": str(target_dir),
        "revision_run_id": review.get("run_id"),
        "human_review_required": True,
    }
    print("\n=== CONTROLLED LIVE TEST RESULT ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (target_dir / "test_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Controlled live Critic->Explanation feedback-loop test")
    parser.add_argument("--confirm-live-test", action="store_true", help="Required acknowledgement: calls real IAedu / ML pipeline")
    parser.add_argument("--patient-id", type=int, default=156906)
    parser.add_argument("--max-revisions", type=int, choices=range(1, 4), metavar="{1,2,3}", default=2)
    args = parser.parse_args()
    if not args.confirm_live_test:
        parser.error("This test uses real IAedu. Add --confirm-live-test to proceed.")
    if args.patient_id <= 0:
        parser.error("--patient-id must be positive")
    result = run_live_test(args.patient_id, args.max_revisions)
    if not result["first_round_rejected"]:
        parser.exit(1, "[FAIL] Critic did not reject the intentionally incorrect first report.\n")
    if not result["real_revision_called"]:
        parser.exit(1, "[FAIL] Orchestrator did not request a real Explanation revision.\n")
    if result["latest_critic_status"] == "approved":
        print("[PASS] The revised explanation was approved by the real Critic.")
    else:
        print("[PARTIAL] The real revision path ran, but approval was not obtained; inspect saved feedback and stop_reason.")


if __name__ == "__main__":
    main()
