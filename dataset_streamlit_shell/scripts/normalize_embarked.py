import json
import pandas as pd

path = "dataset_streamlit_shell/workspace/working.csv"
df = pd.read_csv(path)
before = df["Embarked"].copy()

mapping = {
    "s": "S", "southampton": "S",
    "c": "C", "cherbourg": "C",
    "q": "Q", "queenstown": "Q",
}

raw = df["Embarked"].astype("string").str.strip().str.lower()

def to_code(v):
    if pd.isna(v):
        return pd.NA
    return mapping.get(v, v.upper())

df["Embarked"] = raw.map(to_code).astype("string")

changed = int((before.astype("string").fillna("<NA>") != df["Embarked"].astype("string").fillna("<NA>")).sum())
print("changed rows:", changed)
print(df["Embarked"].value_counts(dropna=False).to_string())

df.to_csv(path, index=False)

log = {
    "created_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "actor": "agent",
    "action": "normalize_embarked",
    "columns": ["Embarked"],
    "rows": changed,
    "note": "將 Embarked 欄位的全名與大小寫寫法統一收斂為 S／C／Q 三種代碼，空值維持不補。",
}
with open("dataset_streamlit_shell/workspace/cleaning_log.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(log, ensure_ascii=False) + "\n")
print("log appended")
