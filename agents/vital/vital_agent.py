"""IACH: Vital Agent. Extraído do notebook de implementação.

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
    for key, value in record.items():

        if pd.isna(value):
            result[key] = None
            continue

        if hasattr(value, "item"):
            value = value.item()

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                result[key] = None
                continue

        result[key] = value

    return result


def numeric(value):
    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    return math.isfinite(float(value))
# 2. Carregamento e verificação dos dados
VITAL_DATA_PATH = locate_csv("vital_agent_data.csv")
vital_df = pd.read_csv(VITAL_DATA_PATH)
if "patientunitstayid" not in vital_df.columns:
    raise ValueError("Falta patientunitstayid.")
if vital_df["patientunitstayid"].duplicated().any():
    raise ValueError("Há IDs duplicados no vital_agent_data.csv.")
vital_df = vital_df.set_index("patientunitstayid", drop=False)
print("Vital:", vital_df.shape, "|", VITAL_DATA_PATH)
# 3. Contratos de saída estruturada
class VitalFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variable: str
    value: float
    flag: Literal["low", "within_reference", "elevated", "unknown"]
    rule_id: str | None = None

class VitalAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: int
    status: Literal["success", "insufficient_data", "error"]
    findings: list[VitalFinding] = Field(default_factory=list)

class VitalRunState(BaseModel):
    patient_id: int
    status: Literal["success", "insufficient_data", "error", "running"]
    raw_data: dict = Field(default_factory=dict)
    range_evaluations: list[dict] = Field(default_factory=list)
    patterns: list[dict] = Field(default_factory=list)
    agent_output: VitalAgentOutput | None = None
    error_message: str | None = None
# 4. Ferramentas determinísticas
def get_vital_data(patientunitstayid: int) -> dict:
    """Dados agregados de um internamento; função determinística, sem chamada ao LLM."""
    if patientunitstayid not in vital_df.index:
        return {"error": f"Patient {patientunitstayid} not found."}
    return clean_record(vital_df.loc[patientunitstayid].to_dict())


# Intervalos ilustrativos de repouso em adultos; NÃO validados para UCI.
# Sem unidades/contexto específicos, deixar flag='unknown' por segurança.
REFERENCE_RANGES = {
    "hr": (60., 100., "PROVISIONAL_REST_HR"),
    "rr": (12., 18., "PROVISIONAL_REST_RR"),
    "sbp": (90., 120., "PROVISIONAL_REST_SBP"),
    "dbp": (60., 80., "PROVISIONAL_REST_DBP"),
    "map": None, "spo2": None,
}
RANGES_VALIDATED = False  # ativar apenas após confirmar critérios + unidades + contexto
VITAL_PREFIXES = ("hr", "rr", "sbp", "dbp", "map", "spo2")
VITAL_STATS = ("first", "mean", "min", "max", "last")


def evaluate_vital_ranges(patient_data: dict) -> list[dict]:
    evaluations = []
    for variable, value in patient_data.items():
        parts = variable.lower().rsplit("_", 1)
        if len(parts) != 2 or parts[0] not in VITAL_PREFIXES or parts[1] not in VITAL_STATS or not numeric(value):
            continue
        rule = REFERENCE_RANGES.get(parts[0])
        flag, rule_id = "unknown", None
        if rule is not None and RANGES_VALIDATED:
            low, high, rule_id = rule
            flag = "low" if value < low else "elevated" if value > high else "within_reference"
        evaluations.append({"variable": variable, "value": float(value), "flag": flag, "rule_id": rule_id})
    return evaluations


def analyse_vital_patterns(patient_data: dict) -> list[dict]:
    patterns = []
    for prefix in VITAL_PREFIXES:
        values = {s: patient_data.get(f"{prefix}_{s}") for s in VITAL_STATS}
        if numeric(values.get("min")) and numeric(values.get("max")):
            patterns.append({"variable": prefix, "pattern_type": "observed_range",
                             "value": float(values["max"] - values["min"]),
                             "source_columns": [f"{prefix}_min", f"{prefix}_max"]})
        if numeric(values.get("last")) and numeric(values.get("mean")):
            patterns.append({"variable": prefix, "pattern_type": "last_minus_mean",
                             "value": float(values["last"] - values["mean"]),
                             "source_columns": [f"{prefix}_last", f"{prefix}_mean"]})
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
def run_vital_agent(patient_id: int) -> VitalRunState:
    patient = get_vital_data(patient_id)
    if "error" in patient:
        return VitalRunState(patient_id=patient_id, status="error", error_message=patient["error"],
            agent_output=VitalAgentOutput(patient_id=patient_id, status="error", findings=[]))

    evaluations = evaluate_vital_ranges(patient)
    patterns = analyse_vital_patterns(patient)
    state = VitalRunState(patient_id=patient_id, status="running", raw_data=patient,
                          range_evaluations=evaluations, patterns=patterns)
    if not evaluations:
        state.status = "insufficient_data"
        state.agent_output = VitalAgentOutput(patient_id=patient_id, status="insufficient_data", findings=[])
        return state

    prompt = f"""You are the Vital Agent in a de-identified ICU research prototype.
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
        output = parse_agent_json(ask_iaedu(prompt), VitalAgentOutput)
        validated_selection(output, patient_id, evaluations)
        state.agent_output = output
        state.status = output.status
    except (requests.RequestException, ValidationError, ValueError, RuntimeError) as exc:
        # Guardar dados determinísticos para o Orchestrator, mas nunca aceitar saída LLM inválida.
        state.status = "error"
        state.error_message = f"{type(exc).__name__}: {str(exc)[:350]}"
        state.agent_output = VitalAgentOutput(patient_id=patient_id, status="error", findings=[])
    return state
