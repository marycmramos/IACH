"""Persistent human-in-the-loop review of an existing IACH pipeline run.

Does not rerun Data Agents, Severity, or the original automatic feedback loop.
Each human revision calls Explanation once and Critic once; a researcher may
request further revisions in subsequent interactions.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Callable, Any

from orchestration import orchestrator as orch


class HumanReviewError(ValueError):
    """User-facing problem with a requested review operation."""


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise HumanReviewError(f"Required file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HumanReviewError(f"Unable to read JSON: {path.name}") from exc
    if not isinstance(data, dict):
        raise HumanReviewError(f"Expected JSON object in {path.name}")
    return data


def _ensure_id(patient_id: int) -> None:
    if isinstance(patient_id, bool) or not isinstance(patient_id, int) or patient_id <= 0:
        raise HumanReviewError("patient_id must be a positive integer")


def _folder(patient_id: int) -> Path:
    # Avoid the side effect of orch.patient_folder creating an empty patient folder.
    return orch.OUTPUT_DIR / str(patient_id)


def _source(patient_id: int) -> tuple[dict, dict, dict, str]:
    folder = _folder(patient_id)
    state = _read_json(folder / "patient_state.json")
    if str(state.get("patientunitstayid")) != str(patient_id):
        raise HumanReviewError("Patient state ID mismatch")
    cycle = state.get("review_cycle") or {}
    run_id = cycle.get("run_id")
    if not isinstance(run_id, str) or not run_id or not all(c.isalnum() or c in "_-" for c in run_id):
        raise HumanReviewError("No valid automatic review run found")
    if not isinstance(state.get("severity"), dict) or state["severity"].get("status") != "success":
        raise HumanReviewError("A successful saved Severity result is required")
    if not isinstance(state.get("explanation"), dict) or state["explanation"].get("status") != "success":
        raise HumanReviewError("A successful saved Explanation result is required")
    critic = state.get("critic")
    if not isinstance(critic, dict) or critic.get("status") not in {"approved", "needs_revision", "review_error"}:
        raise HumanReviewError("A saved Critic result is required")
    if str(critic.get("patientunitstayid")) != str(patient_id):
        raise HumanReviewError("Critic patient ID mismatch")
    if str(state["severity"].get("patientunitstayid")) != str(patient_id):
        raise HumanReviewError("Severity patient ID mismatch")
    if str(state["explanation"].get("patientunitstayid")) != str(patient_id):
        raise HumanReviewError("Explanation patient ID mismatch")
    return state, state["explanation"], critic, run_id


def _session_path(patient_id: int, run_id: str) -> Path:
    return _folder(patient_id) / "human_reviews" / run_id / "human_review.json"


def _store(session: dict, patient_id: int) -> None:
    orch.save_json(_session_path(patient_id, session["run_id"]), session)


def _load_session(patient_id: int) -> dict:
    state, initial, critic, run_id = _source(patient_id)
    path = _session_path(patient_id, run_id)
    if path.exists():
        session = _read_json(path)
        if session.get("patientunitstayid") != patient_id or session.get("run_id") != run_id:
            raise HumanReviewError("Human review session does not match current pipeline run")
        versions = session.get("versions")
        if not isinstance(versions, list) or not versions:
            raise HumanReviewError("Human review session is incomplete")
        return session

    # A session is always pinned to one automatic run. A new orchestrator run
    # creates a new run_id and therefore cannot reuse old human approval.
    if not isinstance(initial.get("explanation"), dict):
        raise HumanReviewError("Saved Explanation report is incomplete")
    session = {
        "patientunitstayid": patient_id,
        "run_id": run_id,
        "status": "pending",
        "current_version": 0,
        "approved_version": None,
        "created_at": orch.now(),
        "updated_at": orch.now(),
        "versions": [{
            "version": 0,
            "source": "automatic_pipeline",
            "explanation": deepcopy(initial),
            "critic": deepcopy(critic),
            "created_at": orch.now(),
        }],
        "decisions": [],
    }
    directory = path.parent
    orch.save_json(directory / "version_00.json", session["versions"][0])
    _store(session, patient_id)
    return session


def get_review(patient_id: int) -> dict:
    """Return existing review session or initialize it from latest pipeline output."""
    _ensure_id(patient_id)
    return _load_session(patient_id)


def current_version(session: dict) -> dict:
    number = session["current_version"]
    matching = [v for v in session["versions"] if v["version"] == number]
    if len(matching) != 1:
        raise HumanReviewError("Current human review version is missing")
    return matching[0]


def approve_review(patient_id: int, reviewer: str = "researcher") -> dict:
    """Explicit human approval, only for a version approved by the Critic."""
    session = get_review(patient_id)
    if session["status"] == "approved":
        raise HumanReviewError("This version is already approved by the investigator")
    version = current_version(session)
    critic = version["critic"]
    if critic.get("status") != "approved" or critic.get("approved") is not True:
        raise HumanReviewError("Cannot approve a version not approved by the Critic")
    reviewer = reviewer.strip()
    if not reviewer or len(reviewer) > 120:
        raise HumanReviewError("A reviewer label (1–120 characters) is required")
    session["status"] = "approved"
    session["approved_version"] = version["version"]
    session["updated_at"] = orch.now()
    session["decisions"].append({
        "type": "approved", "version": version["version"],
        "reviewer": reviewer, "created_at": orch.now(),
    })
    _store(session, patient_id)
    return session


def request_revision(
    patient_id: int,
    feedback: str,
    reviewer: str = "researcher",
    explanation_runner: Callable[..., Any] | None = None,
    critic_runner: Callable[..., Any] | None = None,
) -> dict:
    """Generate one new version from human feedback and check it with Critic.

    An operation is recorded before calling an LLM. If a request fails, keep
    the previous reviewed version unchanged; allow a later retry/new request.
    """
    session = get_review(patient_id)
    if session["status"] == "approved":
        raise HumanReviewError("Approved session is closed; start a new pipeline run to reopen")
    if not isinstance(feedback, str) or not feedback.strip() or len(feedback) > 4000:
        raise HumanReviewError("Feedback must contain 1–4000 characters")
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 120:
        raise HumanReviewError("Reviewer label must contain 1–120 characters")
    feedback = feedback.strip()
    reviewer = reviewer.strip()
    prior = current_version(session)
    next_number = prior["version"] + 1
    if next_number > 9999:
        raise HumanReviewError("Review history limit reached")

    decision = {
        "type": "revision_requested", "version": prior["version"],
        "target_version": next_number, "reviewer": reviewer,
        "feedback": feedback, "result": "pending", "created_at": orch.now(),
    }
    session["decisions"].append(decision)
    session["updated_at"] = orch.now()
    _store(session, patient_id)

    if explanation_runner is None:
        from agents.explanation.explanation_agent import run_explanation_agent
        explanation_runner = run_explanation_agent
    if critic_runner is None:
        from agents.critic.critic_agent import run_critic_agent
        critic_runner = run_critic_agent

    state, _, _, run_id = _source(patient_id)
    # The human session must not silently switch to a different automatic run.
    if run_id != session["run_id"]:
        raise HumanReviewError("Pipeline was rerun; reopen the current human review")
    previous_report = prior["explanation"]["explanation"]
    # Explanation revision_context requires structured issues. Treat human input
    # as untrusted feedback; the Explanation system prompt and evidence prevail.
    issues = [{
        "code": "HUMAN_FEEDBACK", "claim": "Researcher requested changes",
        "reason": feedback, "evidence": "Investigator feedback (not clinical evidence)",
        "suggestion": feedback,
    }]
    explanation_input = {
        "patientunitstayid": patient_id,
        "severity_result": deepcopy(state["severity"]),
        "agent_results": {name: entry.get("data") for name, entry in (state.get("agents") or {}).items()},
    }
    try:
        revised = orch.to_jsonable(explanation_runner(
            explanation_input,
            use_llm=True,
            revision_context={"previous_explanation": deepcopy(previous_report), "issues": issues},
        ))
        if not isinstance(revised, dict) or revised.get("status") != "success":
            raise HumanReviewError("Explanation Agent could not generate the revised report")
        if str(revised.get("patientunitstayid")) != str(patient_id):
            raise HumanReviewError("Explanation Agent returned the wrong patient ID")
        orch.explanation_to_critic_text(revised, patient_id)  # validate completeness
        if revised["explanation"] == previous_report:
            raise HumanReviewError("New explanation is identical to the previous version")
        reviewed = orch.to_jsonable(critic_runner({
            "patientunitstayid": patient_id,
            "severity_result": deepcopy(state["severity"]),
            "explanation_text": orch.explanation_to_critic_text(revised, patient_id),
            "human_feedback": feedback,
            "explanation_report": deepcopy(revised["explanation"]),
            "previous_explanation": deepcopy(previous_report),
        }))
        if not isinstance(reviewed, dict) or reviewed.get("status") not in {
            "approved", "needs_revision", "review_error"
        } or str(reviewed.get("patientunitstayid")) != str(patient_id):
            raise HumanReviewError("Critic Agent returned an invalid review")
        if reviewed.get("approved") is not (reviewed["status"] == "approved"):
            raise HumanReviewError("Critic status and approved flag disagree")
        version = {
            "version": next_number, "source": "human_feedback",
            "explanation": revised, "critic": reviewed,
            "feedback": feedback, "reviewer": reviewer, "created_at": orch.now(),
        }
        orch.save_json(_session_path(patient_id, run_id).parent / f"version_{next_number:02d}.json", version)
        session["versions"].append(version)
        session["current_version"] = next_number
        session["status"] = "pending"
        decision["result"] = "completed"
        decision["completed_at"] = orch.now()
        session["updated_at"] = orch.now()
        _store(session, patient_id)
        return session
    except Exception as exc:
        decision["result"] = "failed"
        decision["error_type"] = type(exc).__name__
        decision["completed_at"] = orch.now()
        session["updated_at"] = orch.now()
        _store(session, patient_id)
        if isinstance(exc, HumanReviewError):
            raise
        # Avoid returning raw API exception messages, which may include secrets.
        raise HumanReviewError(f"Revision failed ({type(exc).__name__}); previous version was preserved") from exc
