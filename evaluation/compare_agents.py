"""Compare A (single agent) and B (multi-agent before Critic).

Run from project root:
    python -u evaluation/compare_agents.py
    python -u evaluation/compare_agents.py --include-train
    python -u evaluation/compare_agents.py --patient-ids 176730 141765

Only checks verifiable structural/numeric/textual properties. It is NOT clinical
validation, and absence of warnings does not establish factual correctness.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATIENTS = ROOT / "outputs" / "patients"
RESULTS = ROOT / "evaluation" / "results"
REPORT_FIELDS = ("mortality_summary", "clinical_summary", "shap_summary", "overall_explanation", "limitations")
PERCENT_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?\s*%")


def _read_result(patient_id: str, name: str) -> dict | None:
    path = PATIENTS / patient_id / f"{name}.json"
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or str(record.get("patientunitstayid")) != patient_id:
        return None
    result = record.get("result")
    if not isinstance(result, dict) or result.get("status") != "success":
        return None
    if str(result.get("patientunitstayid")) != patient_id:
        return None
    return result


def _splits() -> dict[str, str]:
    """Map episode IDs to split via unique patient ID, preserving group split."""
    split_path = ROOT / "ml_model" / "results" / "split_24h.json"
    features_path = ROOT / "ml_model" / "ml_features_24h.csv"
    if not split_path.is_file() or not features_path.is_file():
        return {}
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train = set(map(str, split["train_patient_ids"]))
    test = set(map(str, split["test_patient_ids"]))
    if train & test:
        raise ValueError("Patients appear in both train and test splits")
    mapping = {}
    with features_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pid, episode = row["uniquepid"], row["patientunitstayid"]
            mapping[episode] = "train" if pid in train else "test" if pid in test else "unknown"
    return mapping


def _numeric_probability(severity: dict) -> float:
    probability = (severity.get("mortality_prediction") or {}).get("probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ValueError("Severity probability missing or invalid")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Severity probability out of range")
    return probability


def _expected_pct(probability: float) -> str:
    return f"{probability * 100:.2f}%"


def _inspect(result: dict, severity: dict) -> dict:
    """Evaluate both architectures with exactly the same technical checks."""
    report = result.get("explanation") or {}
    issues = []
    if not isinstance(report, dict):
        report = {}
    fields_ok = all(
        isinstance(report.get(k), list) and bool(report[k]) if k == "limitations"
        else isinstance(report.get(k), str) and bool(report[k].strip())
        for k in REPORT_FIELDS
    )
    if not fields_ok:
        issues.append("Missing or empty report section")

    summary = report.get("mortality_summary") or ""
    prob = _numeric_probability(severity)
    pct = PERCENT_RE.findall(summary) if isinstance(summary, str) else []
    mortality_ok = len(pct) == 1 and pct[0].replace(" ", "").replace(",", ".") == _expected_pct(prob) and summary.strip() == pct[0].strip()
    if not mortality_ok:
        issues.append("Mortality summary does not match XGBoost percentage")

    shap = severity.get("shap_explanation") or {}
    feats = shap.get("top_features") or []
    shap_summary = report.get("shap_summary") or ""
    shap_text = shap_summary.lower() if isinstance(shap_summary, str) else ""
    shap_units_ok = ("log-odds" in shap_text or "log odds" in shap_text or "log_odds" in shap_text) and not bool(PERCENT_RE.search(shap_text))
    if not shap_units_ok:
        issues.append("SHAP units missing or SHAP section contains percentages")
    feature_mentions = sum(bool(re.search(r"(?<!\w)" + re.escape(str(x.get("feature", ""))) + r"(?!\w)", shap_summary, flags=re.IGNORECASE)) for x in feats if x.get("feature"))
    missing_imputation_disclosure = any(x.get("was_imputed") is True for x in feats) and not any(
        word in (shap_text + " " + " ".join(map(str, report.get("limitations", []))).lower())
        for word in ("imput", "missing")
    )
    imputation_ok = not missing_imputation_disclosure
    if not imputation_ok:
        issues.append("Imputed SHAP features not disclosed")

    apache = severity.get("apache_comparison") or {}
    overall = report.get("overall_explanation") or ""
    overall_text = overall.lower() if isinstance(overall, str) else ""
    versions = (apache.get("versions") or {}) if apache.get("status") == "descriptive" else {}
    reference_versions = {key: ver for key, ver in versions.items() if isinstance(ver, dict) and ver.get("status") == "descriptive"}
    apache_ok = True
    if reference_versions:
        for key, ver in reference_versions.items():
            val = ver.get("apache_probability")
            if not isinstance(val, (int, float)) or not math.isfinite(val) or _expected_pct(val) not in overall:
                apache_ok = False
            # IV should not be mistaken for IVa: verify each version name appears.
            label_re = r"\bapache\s*iv\s*a\b" if key == "IVa" else r"\bapache\s*iv\b(?!\s*a)"
            if not re.search(label_re, overall_text):
                apache_ok = False
    else:
        apache_ok = "apache" in overall_text and any(w in overall_text for w in ("unavailable", "not supplied", "not available", "no ", "cannot", "could not"))
    if not apache_ok:
        issues.append("APACHE comparison missing/inconsistent with reference")

    return {
        "sections_ok": fields_ok,
        "mortality_ok": mortality_ok,
        "shap_units_ok": shap_units_ok,
        "shap_features_mentioned": feature_mentions,
        "shap_features_supplied": len(feats),
        "imputation_disclosed": imputation_ok,
        "apache_ok": apache_ok,
        "technical_issue_count": len(issues),
        "technical_issues": " | ".join(issues),
        "llm_attempts": result.get("llm_attempts", ""),
    }


def compare(patient_id: str, split: str) -> list[dict]:
    severity = _read_result(patient_id, "severity")
    a = _read_result(patient_id, "single_agent")
    b = _read_result(patient_id, "explanation")
    if severity is None or a is None or b is None:
        print(f"[SKIP] {patient_id}: missing successful " + ", ".join(
            name for name, item in (("severity", severity), ("single_agent", a), ("explanation", b)) if item is None
        ))
        return []
    try:
        _numeric_probability(severity)
    except ValueError as exc:
        print(f"[SKIP] {patient_id}: {exc}")
        return []
    results = []
    for arm, obj in (("A_single", a), ("B_multi_no_critic", b)):
        # Ensure both output probabilities are the same numerical reference.
        output_prob = (obj.get("mortality_prediction") or {}).get("probability")
        checks = _inspect(obj, severity)
        consistent_reference = isinstance(output_prob, (int, float)) and not isinstance(output_prob, bool) and math.isclose(output_prob, _numeric_probability(severity), abs_tol=1e-10, rel_tol=0)
        if not consistent_reference:
            checks["technical_issue_count"] += 1
            checks["technical_issues"] += (" | " if checks["technical_issues"] else "") + "Output probability differs from Severity"
        results.append({"episode_id": patient_id, "split": split, "architecture": arm,
                        "xgboost_probability": _numeric_probability(severity), **checks})
    print(f"[OK] {patient_id}: A issues={results[0]['technical_issue_count']}; B issues={results[1]['technical_issue_count']}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-train", action="store_true", help="Include training episodes (exploratory only)")
    parser.add_argument("--patient-ids", nargs="+", help="Compare only these episodes")
    args = parser.parse_args()
    splits = _splits()
    if not splits:
        raise SystemExit("Split files missing; cannot confirm independent test episodes.")
    ids = args.patient_ids or sorted(p.name for p in PATIENTS.iterdir() if p.is_dir())
    rows = []
    for patient_id in ids:
        split = splits.get(str(patient_id), "not_in_ml_dataset")
        if split != "test" and not args.include_train:
            print(f"[SKIP] {patient_id}: {split}; default comparison uses held-out test only")
            continue
        rows.extend(compare(str(patient_id), split))
    if not rows:
        print("No complete A/B pairs yet. Run the single-agent baseline for eligible test episodes.")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "comparison_A_B.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCompared {len(rows) // 2} episodes ({len(rows)} architecture outputs)")
    print(f"CSV saved: {output}")
    print("Note: checks are technical only, not clinical validation or proof of architectural superiority.")


if __name__ == "__main__":
    main()
