"""IACH: Laboratory Agent. Extraído do notebook de implementação.

Os testes locais do notebook não são executados ao importar este módulo.
"""

# 1. Configuração IAedu e funções partilhadas
import os, json, uuid, re, math
from pathlib import Path
from typing import Literal
import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict, ValidationError

# Procura .env na pasta do notebook ou em pastas superiores.
# Caminho absoluto para a raiz do projeto, independentemente do diretório atual.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)
API_ENDPOINT = os.getenv("IAEDU_API_ENDPOINT", "").strip()
CHANNEL_ID = os.getenv("IAEDU_CHANNEL_ID", "").strip()
API_KEY = os.getenv("IAEDU_API_KEY", "").strip()
if not all((API_ENDPOINT, CHANNEL_ID, API_KEY)):
    raise RuntimeError("Configuração IAedu em falta. Cria .env com IAEDU_API_ENDPOINT, IAEDU_CHANNEL_ID e IAEDU_API_KEY.")
if not API_ENDPOINT.startswith("https://"):
    raise ValueError("Usa o endpoint IAedu HTTPS completo, terminado em /stream (conforme o teu serviço).")
print("IAedu carregada do .env (credenciais ocultas).")

def locate_csv(filename):
    paths = [PROJECT_ROOT / "processed_data" / filename,
             PROJECT_ROOT / "data" / "processed_data" / filename,
             PROJECT_ROOT / "data" / filename,
             Path.cwd() / "processed_data" / filename,
             Path.cwd() / "data" / "processed_data" / filename]
    for path in paths:
        if path.is_file(): return path
    raise FileNotFoundError(f"Não encontrei {filename}. Procurei: {[str(p) for p in paths]}")

def clean_record(record):
    result = {}
    for k, v in record.items():
        if pd.isna(v): result[k] = None
        elif hasattr(v, "item"): result[k] = v.item()
        else: result[k] = v
        if isinstance(result[k], float) and not math.isfinite(result[k]):
            result[k] = None
    return result

def numeric(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))
# 2. Carregamento e verificação dos dados
LAB_DATA_PATH = locate_csv("lab_agent_data.csv")
LAB_RAW_PATH = locate_csv("lab_agent_raw.csv")
lab_df = pd.read_csv(LAB_DATA_PATH)
lab_raw_df = pd.read_csv(LAB_RAW_PATH)
LABS = ("BUN", "creatinine", "glucose", "potassium", "sodium")
LAB_STATS = ("first", "mean", "min", "max", "last")
required_summary = {"patientunitstayid"} | {f"{lab}_{s}" for lab in LABS for s in LAB_STATS}
required_raw = {"patientunitstayid", "labname", "labresult", "labresultoffset"}
if required_summary - set(lab_df.columns):
    raise ValueError(f"Colunas agregadas em falta: {sorted(required_summary-set(lab_df.columns))}")
if required_raw - set(lab_raw_df.columns):
    raise ValueError(f"Colunas raw em falta: {sorted(required_raw-set(lab_raw_df.columns))}")
if lab_df["patientunitstayid"].duplicated().any():
    raise ValueError("Há patientunitstayid duplicados no lab_agent_data.csv.")
lab_df = lab_df.set_index("patientunitstayid", drop=False)
lab_raw_df = lab_raw_df.sort_values(["patientunitstayid", "labname", "labresultoffset"])
print("Lab agregado:", lab_df.shape, "|", LAB_DATA_PATH)
print("Lab temporal:", lab_raw_df.shape, "|", LAB_RAW_PATH)
# 3. Contratos de saída estruturada
class LabFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variable: str
    value: float
    flag: Literal["low", "within_reference", "elevated", "unknown"]
    rule_id: str | None = None

class LabAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: int
    status: Literal["success", "insufficient_data", "error"]
    findings: list[LabFinding] = Field(default_factory=list)

class LabRunState(BaseModel):
    patient_id: int
    status: Literal["success", "insufficient_data", "error", "running"]
    raw_data: dict = Field(default_factory=dict)
    range_evaluations: list[dict] = Field(default_factory=list)
    patterns: list[dict] = Field(default_factory=list)
    agent_output: LabAgentOutput | None = None
    error_message: str | None = None
# 4. Ferramentas determinísticas
def get_lab_data(patientunitstayid: int) -> dict:
    """Consulta estatísticas laboratoriais de um internamento."""
    if patientunitstayid not in lab_df.index:
        return {"error": f"Patient {patientunitstayid} not found."}
    return clean_record(lab_df.loc[patientunitstayid].to_dict())

# Intervalos ilustrativos, NÃO certificados para estes CSVs.
LAB_REFERENCE_RANGES = {
    "BUN": (7., 20., "PROVISIONAL_BUN_MG_DL"),
    "creatinine": (0.6, 1.3, "PROVISIONAL_CREATININE_MG_DL"),
    "glucose": (70., 140., "PROVISIONAL_GLUCOSE_MG_DL"),
    "potassium": (3.5, 5.0, "PROVISIONAL_POTASSIUM_MMOL_L"),
    "sodium": (135., 145., "PROVISIONAL_SODIUM_MMOL_L"),
}
LAB_RANGES_VALIDATED = False  # confirmar unidades + limites locais antes de habilitar flags

def evaluate_lab_ranges(patient_data: dict) -> list[dict]:
    evaluations = []
    for lab in LABS:
        for stat in LAB_STATS:
            variable = f"{lab}_{stat}"
            value = patient_data.get(variable)
            if not numeric(value): continue
            low, high, rule_id = LAB_REFERENCE_RANGES[lab]
            flag = ("low" if value < low else "elevated" if value > high else "within_reference") if LAB_RANGES_VALIDATED else "unknown"
            evaluations.append({"variable": variable, "value": float(value),
                                "flag": flag, "rule_id": rule_id if LAB_RANGES_VALIDATED else None})
    return evaluations


def analyse_lab_patterns(patient_id: int, patient_data: dict) -> list[dict]:
    """
    Analyse laboratory changes using measurement timestamps.

    Measurements at identical timestamps are averaged
    only for first-to-last comparisons.

    A first-to-last difference does not establish
    a sustained clinical trend.
    """

    subset = lab_raw_df.loc[
        lab_raw_df["patientunitstayid"] == patient_id
    ]

    patterns = []

    for lab in LABS:

        vals = subset.loc[
            subset["labname"] == lab,
            ["labresultoffset", "labresult"]
        ].dropna().copy()

        if vals.empty:
            continue

        # Ensure numeric values
        vals["labresultoffset"] = pd.to_numeric(
            vals["labresultoffset"], errors="coerce"
        )
        vals["labresult"] = pd.to_numeric(
            vals["labresult"], errors="coerce"
        )

        vals = vals.dropna()

        if vals.empty:
            continue

        # Count timestamps containing multiple measurements
        grouped = vals.groupby("labresultoffset")["labresult"]

        counts = grouped.size()
        distinct_values = grouped.nunique()

        repeated_timestamps = int((counts > 1).sum())
        conflicting_timestamps = int(
            (distinct_values > 1).sum()
        )

        # Average measurements recorded at the same minute
        timeline = (
            grouped.mean()
            .reset_index()
            .sort_values("labresultoffset")
        )

        n_timepoints = len(timeline)

        first = float(timeline.iloc[0]["labresult"])
        last = float(timeline.iloc[-1]["labresult"])

        # One timepoint is not sufficient to assess change
        if n_timepoints < 2:
            change = None
            change_status = "insufficient_timepoints"

        else:
            change = float(last - first)

            if math.isclose(change, 0, abs_tol=1e-9):
                change_status = "no_net_change"

            elif change > 0:
                change_status = "increased"

            else:
                change_status = "decreased"

        patterns.append({
            "variable": lab,
            "pattern_type": "first_to_last_change",
            "value": change,
            "change_status": change_status,
            "first_value": first,
            "last_value": last,
            "first_offset_minutes": float(
                timeline.iloc[0]["labresultoffset"]
            ),
            "last_offset_minutes": float(
                timeline.iloc[-1]["labresultoffset"]
            ),
            "distinct_timepoints": n_timepoints,
            "raw_measurements": len(vals),
            "repeated_timestamps": repeated_timestamps,
            "conflicting_timestamps": conflicting_timestamps,
            "source_columns": [
                "labresultoffset",
                "labresult"
            ],
            "interpretation_limit": (
                "First-to-last change only; "
                "not evidence of a sustained trend."
            )
        })

    return patterns
# 5. Cliente IAedu e validação das respostas
# IAedu comunica por eventos de streaming (SSE). Não enviar informações identificáveis.
def ask_iaedu(message: str) -> str:
    """Envia uma mensagem à IAedu e interpreta o streaming."""

    headers = {"x-api-key": API_KEY}

    fields = {
        "channel_id": (None, CHANNEL_ID),
        "thread_id": (None, str(uuid.uuid4())),
        "user_info": (None, "{}"),
        "message": (None, message),
    }

    chunks = []
    final_message = None

    with requests.post(
        API_ENDPOINT,
        headers=headers,
        files=fields,
        stream=True,
        timeout=(15, 120)
    ) as response:

        response.raise_for_status()

        for raw in response.iter_lines(decode_unicode=True):

            if not raw:
                continue

            line = (
                raw if isinstance(raw, str)
                else raw.decode("utf-8", errors="replace")
            ).strip()

            # Ignorar metadados SSE
            if line.startswith(":") or line.startswith("event:"):
                continue

            # Aceitar JSON direto e SSE com prefixo data:
            if line.startswith("data:"):
                line = line[5:].strip()

            if line == "[DONE]":
                break

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(obj, dict):
                continue

            event_type = obj.get("type")

            if event_type == "error":
                raise RuntimeError(
                    f"IAedu error: {str(obj.get('content', obj.get('message', 'unknown')))[:300]}"
                )

            elif event_type == "token":
                content = obj.get("content")

                if isinstance(content, str):
                    chunks.append(content)

            elif event_type == "message":
                content = obj.get("content")

                if isinstance(content, dict):
                    text = content.get("content")

                    if isinstance(text, str):
                        final_message = text

            elif isinstance(obj.get("choices"), list):
                for choice in obj["choices"]:
                    delta = choice.get("delta", {})
                    if isinstance(delta, dict):
                        text = delta.get("content")
                        if isinstance(text, str):
                            chunks.append(text)

    answer = (
        final_message
        if final_message is not None
        else "".join(chunks)
    ).strip()

    if not answer:
        raise ValueError(
            "A IAedu devolveu uma resposta vazia "
            "ou um formato de streaming não reconhecido."
        )
    return answer

def parse_agent_json(reply: str, output_model):
    content = reply.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.I)
        content = re.sub(r"\s*```$", "", content).strip()
    return output_model.model_validate_json(content)
def validated_selection(output, patient_id, evaluations):
    if output.patient_id != patient_id or output.status == "error":
        raise ValueError("ID ou estado devolvido pelo LLM inválido.")
    evidence = {row["variable"]: row for row in evaluations}
    selected = set()
    for finding in output.findings:
        if finding.variable in selected:
            raise ValueError(f"Finding duplicado: {finding.variable}")
        selected.add(finding.variable)
        source = evidence.get(finding.variable)
        if source is None:
            raise ValueError(f"Finding não suportado: {finding.variable}")
        if (not math.isclose(finding.value, source["value"], rel_tol=1e-9, abs_tol=1e-6)
            or finding.flag != source["flag"] or finding.rule_id != source["rule_id"]):
            raise ValueError(f"Finding não corresponde aos dados: {finding.variable}")
    if output.status == "insufficient_data" and output.findings:
        raise ValueError("insufficient_data não pode conter findings.")
# 6. Execução do agente
def run_lab_agent(patient_id: int) -> LabRunState:
    patient = get_lab_data(patient_id)
    if "error" in patient:
        return LabRunState(patient_id=patient_id, status="error", error_message=patient["error"],
            agent_output=LabAgentOutput(patient_id=patient_id, status="error", findings=[]))

    evaluations = evaluate_lab_ranges(patient)
    patterns = analyse_lab_patterns(patient_id, patient)
    state = LabRunState(patient_id=patient_id, status="running", raw_data=patient,
                          range_evaluations=evaluations, patterns=patterns)
    if not evaluations:
        state.status = "insufficient_data"
        state.agent_output = LabAgentOutput(patient_id=patient_id, status="insufficient_data", findings=[])
        return state

    prompt = f"""You are the Lab Agent in a de-identified ICU research prototype.
Only select entries from the deterministic PRECOMPUTED evaluations. Do not recalculate,
reclassify, invent observations, diagnose conditions, predict mortality or recommend treatment.
Flags are 'unknown' until approved unit-specific, context-specific ranges are installed.
Give priority to interesting observations, but never turn 'unknown' into 'elevated' or 'low'.
Patterns are contextual numeric summaries, not evidence of persistent or causal trends.
Return ONLY a JSON object with exactly patient_id, status, findings (no Markdown).
Each finding must have exactly variable, value, flag, rule_id.
Status must be success or insufficient_data. Empty findings are allowed.
Patient ID: {patient_id}
Precomputed evaluations: {json.dumps(evaluations, ensure_ascii=False, allow_nan=False)}
Context patterns (NOT findings): {json.dumps(patterns, ensure_ascii=False, allow_nan=False)}
Required shape: {json.dumps({'patient_id': patient_id, 'status': 'success', 'findings': []})}"""

    try:
        output = parse_agent_json(ask_iaedu(prompt), LabAgentOutput)
        validated_selection(output, patient_id, evaluations)
        state.agent_output = output
        state.status = output.status
    except (requests.RequestException, ValidationError, ValueError, RuntimeError) as exc:
        # Guardar dados determinísticos para o Orchestrator, mas nunca aceitar saída LLM inválida.
        state.status = "error"
        state.error_message = f"{type(exc).__name__}: {str(exc)[:350]}"
        state.agent_output = LabAgentOutput(patient_id=patient_id, status="error", findings=[])
    return state
