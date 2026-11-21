
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    ROOT / "ml_model/ml_features_24h.csv",
    dtype={"patientunitstayid": str, "uniquepid": str}
)

with open(
    ROOT / "ml_model/results/split_24h.json",
    encoding="utf-8"
) as f:
    split = json.load(f)

train_ids = set(map(str, split["train_patient_ids"]))
test_ids = set(map(str, split["test_patient_ids"]))

df["split"] = df["uniquepid"].map(
    lambda pid: (
        "train" if pid in train_ids
        else "test" if pid in test_ids
        else "unknown"
    )
)

if (df["split"] == "unknown").any():
    raise ValueError("Existem pacientes sem split atribuído.")

if set(train_ids) & set(test_ids):
    raise ValueError("Existem pacientes nos dois conjuntos.")

output = ROOT / "evaluation/results"
output.mkdir(parents=True, exist_ok=True)

df[["patientunitstayid", "uniquepid", "target_icu", "split"]].to_csv(
    output / "patient_splits.csv",
    index=False
)

print("Distribuição dos episódios:")
print(df.groupby("split")["target_icu"].agg(
    episodes="size",
    deaths="sum"
))

print("\nVerificação de IDs:")
for episode_id in ["176730", "148611", "156906", "141765", "1694608"]:
    row = df[df["patientunitstayid"] == episode_id]

    if row.empty:
        print(f"{episode_id}: não existe no dataset ML")
    else:
        print(
            f"{episode_id}: {row.iloc[0]['split']} | "
            f"outcome={row.iloc[0]['target_icu']}"
        )

print("\nFicheiro criado: evaluation/results/patient_splits.csv")