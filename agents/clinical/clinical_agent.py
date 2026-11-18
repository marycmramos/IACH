"""IACH: Clinical Agent. Extraído do notebook de implementação.

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


def clean_scalar(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        return value if value else None

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def parse_list_field(value):
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        ]

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        item.strip()
        for item in value.split(";")
        if item.strip()
    ]
# 2. Carregamento e verificação dos dados
CLINICAL_DATA_PATH = locate_csv("clinical_agent_data.csv")
clinical_df = pd.read_csv(CLINICAL_DATA_PATH)
CLINICAL_COLUMNS = ("age","gender","unit_type","admission_source","admission_diagnosis","diagnoses_0_24h","past_history")

required_columns = {"patientunitstayid",*CLINICAL_COLUMNS}

missing_columns = (required_columns- set(clinical_df.columns))

if missing_columns:
    raise ValueError("Colunas em falta: "
        f"{sorted(missing_columns)}")

if clinical_df["patientunitstayid"].duplicated().any():

    raise ValueError(
        "Há patientunitstayid duplicados "
        "no clinical_agent_data.csv."
    )

clinical_df = clinical_df.set_index("patientunitstayid",drop=False)

print("Clinical dataset:",clinical_df.shape, "|",CLINICAL_DATA_PATH)
# 3. Contratos de saída estruturada
class ClinicalFinding(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    category: Literal[
        "age",
        "gender",
        "unit_type",
        "admission_source",
        "admission_diagnosis",
        "diagnosis_0_24h",
        "past_history"
    ]

    value: str
    
class ClinicalAgentOutput(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    patient_id: int

    status: Literal[
        "success",
        "insufficient_data",
        "error"
    ]

    findings: list[
        ClinicalFinding
    ] = Field(
        default_factory=list
    )
    
class ClinicalRunState(BaseModel):

    patient_id: int

    status: Literal[
        "success",
        "insufficient_data",
        "error",
        "running"
    ]

    raw_data: dict = Field(
        default_factory=dict
    )

    evidence: list[dict] = Field(
        default_factory=list
    )

    agent_output: (
        ClinicalAgentOutput | None
    ) = None

    error_message: (
        str | None
    ) = None
    

# 4. Ferramentas determinísticas
def get_clinical_data(
    patientunitstayid: int
) -> dict:
    """
    Consulta os dados clínicos estruturados
    de um internamento na ICU.
    """

    if patientunitstayid not in clinical_df.index:

        return {
            "error":
            f"Patient {patientunitstayid} not found."
        }


    row = clinical_df.loc[
        patientunitstayid
    ]


    return {
        "patientunitstayid":
            int(patientunitstayid),

        "age":
            clean_scalar(
                row["age"]
            ),

        "gender":
            clean_scalar(
                row["gender"]
            ),

        "unit_type":
            clean_scalar(
                row["unit_type"]
            ),

        "admission_source":
            clean_scalar(
                row["admission_source"]
            ),

        "admission_diagnosis":
            clean_scalar(
                row["admission_diagnosis"]
            ),

        "diagnoses_0_24h":
            parse_list_field(
                row["diagnoses_0_24h"]
            ),

        "past_history":
            parse_list_field(
                row["past_history"]
            )
    }
    
def build_clinical_evidence(
    patient_data: dict
) -> list[dict]:
    """
    Builds the exact set of clinical facts that
    the LLM is allowed to select as findings.
    """

    evidence = []


    age = patient_data.get("age")

    if age is not None:

        # Avoid displaying integer ages as 63.0
        if isinstance(age, float) and age.is_integer():
            age = int(age)

        evidence.append({
            "category": "age",
            "value": str(age)
        })


    gender = patient_data.get("gender")

    if gender:

        evidence.append({
            "category": "gender",
            "value": str(gender)
        })


    unit_type = patient_data.get(
        "unit_type"
    )

    if unit_type:

        evidence.append({
            "category": "unit_type",
            "value": str(unit_type)
        })


    admission_source = patient_data.get(
        "admission_source"
    )

    if admission_source:

        evidence.append({
            "category": "admission_source",
            "value": str(admission_source)
        })


    admission_diagnosis = patient_data.get(
        "admission_diagnosis"
    )

    if admission_diagnosis:

        evidence.append({
            "category": "admission_diagnosis",
            "value": str(admission_diagnosis)
        })


    for diagnosis in patient_data.get(
        "diagnoses_0_24h",
        []
    ):

        evidence.append({
            "category": "diagnosis_0_24h",
            "value": str(diagnosis)
        })


    for history in patient_data.get(
        "past_history",
        []
    ):

        evidence.append({
            "category": "past_history",
            "value": str(history)
        })


    return evidence
# 5. Cliente IAedu e validação das respostas
# IAedu comunica por eventos de streaming (SSE).
# Não enviar informações identificáveis.

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

def parse_agent_json(
    reply: str,
    output_model
):

    content = reply.strip()


    if content.startswith("```"):

        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content,
            flags=re.I
        )

        content = re.sub(
            r"\s*```$",
            "",
            content
        ).strip()


    return output_model.model_validate_json(
        content
    )
def validated_selection(
    output,
    patient_id,
    evidence
):

    if (
        output.patient_id != patient_id
        or output.status == "error"
    ):

        raise ValueError(
            "ID ou estado devolvido "
            "pelo LLM inválido."
        )


    allowed = {
        (
            row["category"],
            row["value"]
        )
        for row in evidence
    }


    selected = set()


    for finding in output.findings:

        key = (
            finding.category,
            finding.value
        )


        if key in selected:

            raise ValueError(
                f"Finding duplicado: {key}"
            )


        selected.add(key)


        if key not in allowed:

            raise ValueError(
                "Finding não suportado "
                f"pelos dados: {key}"
            )


    if (
        output.status == "insufficient_data"
        and output.findings
    ):

        raise ValueError(
            "insufficient_data não pode "
            "conter findings."
        )
# 6. Execução do agente
def run_clinical_agent(
    patient_id: int
) -> ClinicalRunState:


    patient = get_clinical_data(
        patient_id
    )


    if "error" in patient:

        return ClinicalRunState(
            patient_id=patient_id,
            status="error",

            error_message=patient[
                "error"
            ],

            agent_output=(
                ClinicalAgentOutput(
                    patient_id=patient_id,
                    status="error",
                    findings=[]
                )
            )
        )


    evidence = build_clinical_evidence(
        patient
    )


    state = ClinicalRunState(
        patient_id=patient_id,
        status="running",
        raw_data=patient,
        evidence=evidence
    )


    if not evidence:

        state.status = (
            "insufficient_data"
        )

        state.agent_output = (
            ClinicalAgentOutput(
                patient_id=patient_id,
                status="insufficient_data",
                findings=[]
            )
        )

        return state


    prompt = f"""
You are the Clinical Agent in a de-identified ICU research prototype.

Only select entries from the deterministic PROVIDED EVIDENCE.

Do not:
- invent observations;
- modify or rephrase evidence;
- add diagnoses;
- infer diagnoses that are not explicitly present;
- calculate a severity score;
- predict mortality;
- recommend treatment;
- provide medical advice.

Your role is only to select the most informative clinical facts already
present in the evidence.

Give priority to:
1. admission diagnosis;
2. diagnoses documented during the first 24 hours;
3. relevant past medical history.

Age, gender, ICU type and admission source may also be selected when useful
for describing the patient's clinical context.

Every finding must correspond EXACTLY to one item in PROVIDED EVIDENCE.

Return ONLY a JSON object with exactly:
patient_id, status, findings

Each finding must contain exactly:
category, value

Allowed categories:
age
gender
unit_type
admission_source
admission_diagnosis
diagnosis_0_24h
past_history

Status must be success or insufficient_data.

Empty findings are allowed.

Patient ID:
{patient_id}

PROVIDED EVIDENCE:
{json.dumps(
    evidence,
    ensure_ascii=False,
    allow_nan=False
)}

Required shape:
{json.dumps(
    {
        "patient_id": patient_id,
        "status": "success",
        "findings": []
    }
)}
"""


    try:

        output = parse_agent_json(
            ask_iaedu(prompt),
            ClinicalAgentOutput
        )


        validated_selection(
            output,
            patient_id,
            evidence
        )


        state.agent_output = output

        state.status = output.status


    except (
        requests.RequestException,
        ValidationError,
        ValueError,
        RuntimeError
    ) as exc:

        # Guardar dados determinísticos para o
        # Orchestrator, mas nunca aceitar saída
        # LLM inválida.

        state.status = "error"

        state.error_message = (
            f"{type(exc).__name__}: "
            f"{str(exc)[:350]}"
        )

        state.agent_output = (
            ClinicalAgentOutput(
                patient_id=patient_id,
                status="error",
                findings=[]
            )
        )


    return state
