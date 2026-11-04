import json
import pandas as pd

path = "dataset_streamlit_shell/workspace/working.csv"
df = pd.read_csv(path)
before_n = len(df)

# 以 Name 分組，每組保留「空值最少」的那筆；空值數相同則保留原順序第一筆
df["_na"] = df.isna().sum(axis=1)
df["_ord"] = range(len(df))
keep = (
    df.sort_values(["_na", "_ord"])
      .drop_duplicates(subset=["Name"], keep="first")
      .sort_values("_ord")
)
removed = before_n - len(keep)
kept_ids = keep["PassengerId"].tolist()

df = keep.drop(columns=["_na", "_ord"]).reset_index(drop=True)

print("removed rows:", removed)
print("new shape:", df.shape)
print("保留的 PassengerId:", kept_ids)
print("Name 重複檢查:", int(df["Name"].duplicated().sum()))

df.to_csv(path, index=False)

log = {
    "created_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "actor": "agent",
    "action": "drop_duplicate_names",
    "columns": ["Name"],
    "rows": removed,
    "note": "以 Name 判斷重複，每組保留空值最少的一筆，刪除其餘重複列。",
}
with open("dataset_streamlit_shell/workspace/cleaning_log.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(log, ensure_ascii=False) + "\n")
print("log appended")
