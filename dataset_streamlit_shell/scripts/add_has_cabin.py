"""新增『是否有船艙』欄位：船艙編號有值 -> 1，空值 -> 0。"""
import json
from datetime import datetime

import pandas as pd

WORKING = "dataset_streamlit_shell/workspace/working.csv"
LOG = "dataset_streamlit_shell/workspace/cleaning_log.jsonl"
NEW_COL = "是否有船艙"

df = pd.read_csv(WORKING)
before_cols = df.shape[1]

cabin = df["船艙編號"].astype("string").str.strip()
df[NEW_COL] = cabin.notna() & cabin.ne("")

rows_affected = int(df.shape[0])

df.to_csv(WORKING, index=False, encoding="utf-8")

entry = {
    "created_at": datetime.now().replace(microsecond=0).isoformat(),
    "actor": "agent",
    "action": "add_has_cabin_flag",
    "columns": [NEW_COL],
    "rows": rows_affected,
    "note": "依船艙編號是否有值新增是否有船艙旗標欄位（1=有、0=無）。",
}
with open(LOG, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("欄位數:", before_cols, "->", df.shape[1])
print(df[NEW_COL].value_counts().to_dict())
