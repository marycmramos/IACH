"""Auditoria offline dos outputs dos agentes. Executar na raiz do projeto:
python -u evaluation/evaluate_agents.py
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATIENTS = ROOT / 'outputs' / 'patients'
OUTPUT = ROOT / 'evaluation' / 'results'
STAGES = ('vital', 'laboratory', 'clinical', 'severity', 'explanation')
FIELDS = ('mortality_summary', 'clinical_summary', 'shap_summary', 'overall_explanation')


def load_stage(folder, stage):
    path = folder / (stage + '.json')
    if not path.exists():
        return None, None
    raw = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        return raw, None
    payload = raw.get('result' if stage in ('severity', 'explanation') else 'data')
    return raw, payload if isinstance(payload, dict) else None


def equal_numbers(a, b, tolerance=1e-6):
    return (isinstance(a, (int, float)) and not isinstance(a, bool)
            and isinstance(b, (int, float)) and not isinstance(b, bool)
            and math.isfinite(a) and math.isfinite(b)
            and abs(a-b) <= tolerance)


def audit(folder):
    records = {stage: load_stage(folder, stage) for stage in STAGES}
    issues = []
    for stage, (wrapper, data) in records.items():
        if wrapper is None:
            issues.append(stage + ': ficheiro ausente')
        elif data is None:
            issues.append(stage + ': estrutura nao reconhecida')
        elif data.get('status') != 'success':
            issues.append(stage + ': status=' + str(data.get('status')))
    sev = records['severity'][1] or {}
    exp = records['explanation'][1] or {}
    sev_pred = sev.get('mortality_prediction') or {}
    exp_pred = exp.get('mortality_prediction') or {}
    p = sev_pred.get('probability')
    report = exp.get('explanation') or {}
    if sev and exp:
        if not equal_numbers(p, exp_pred.get('probability')):
            issues.append('probabilidade diferente entre Severity e Explanation')
        sev_features = (sev.get('shap_explanation') or {}).get('top_features') or []
        exp_features = (exp.get('shap_explanation') or {}).get('top_features') or []
        if len(sev_features) != len(exp_features) or any(
            a.get('feature') != b.get('feature') or
            not equal_numbers(a.get('shap_value'), b.get('shap_value'))
            for a, b in zip(sev_features, exp_features)
        ):
            issues.append('contribuicoes SHAP diferentes entre Severity e Explanation')
    if exp:
        for key in FIELDS:
            if not isinstance(report.get(key), str) or not report[key].strip():
                issues.append('explicacao: falta ' + key)
        if not isinstance(report.get('limitations'), list):
            issues.append('explicacao: limitations nao e lista')
    row = {'episode_id': folder.name,
           **{stage+'_file': records[stage][0] is not None for stage in STAGES},
           'model_probability': p,
           'has_final_explanation': all(isinstance(report.get(key), str) and report[key].strip() for key in FIELDS),
           'automated_issue_count': len(issues), 'automated_issues': ' | '.join(issues),
           'manual_factual_consistency': '', 'manual_shap_alignment': '',
           'manual_completeness': '', 'manual_clarity': '', 'manual_evidence': ''}
    detail = {'episode_id': folder.name, 'issues': issues,
              'mortality_probability': p, 'explanation': report or None}
    return row, detail


def main():
    if not PATIENTS.is_dir():
        raise SystemExit('Pasta nao encontrada: ' + str(PATIENTS))
    folders = sorted(p for p in PATIENTS.iterdir() if p.is_dir())
    if not folders:
        raise SystemExit('Nao existem episodios em ' + str(PATIENTS))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, details = [], []
    for folder in folders:
        row, detail = audit(folder)
        rows.append(row)
        details.append(detail)
        print(f"Episodio {row['episode_id']}: explicacao={row['has_final_explanation']}; avisos={row['automated_issue_count']}")
    with (OUTPUT / 'agents_review.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / 'agents_audit.json').write_text(json.dumps(
        {'note': 'Auditoria tecnica, nao validacao clinica.', 'episodes': details},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nEpisodios encontrados: {len(rows)}")
    print(f"Explicacoes finais: {sum(row['has_final_explanation'] for row in rows)}")
    print('Ficheiros gerados em: ' + str(OUTPUT))


if __name__ == '__main__':
    main()
