import json
import pandas as pd

path = "dataset_streamlit_shell/workspace/working.csv"
df = pd.read_csv(path)
before = df["Sex"].copy()

raw = df["Sex"].astype("string").str.strip().str.lower()

male_set = {"male", "m", "man"}
female_set = {"female", "f", "woman"}

df["Sex"] = raw.map(lambda v: "male" if v in male_set else ("female" if v in female_set else v))

changed = int((before.astype("string") != df["Sex"].astype("string")).sum())
print("changed rows:", changed)
print(df["Sex"].value_counts(dropna=False).to_string())

df.to_csv(path, index=False)

log = {
    "created_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "actor": "agent",
    "action": "normalize_sex",
    "columns": ["Sex"],
    "rows": changed,
    "note": "將 Sex 欄位的多種寫法（M/Male/male/man 等）統一收斂為 male／female 兩種值。",
}
with open("dataset_streamlit_shell/workspace/cleaning_log.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(log, ensure_ascii=False) + "\n")
print("log appended")
