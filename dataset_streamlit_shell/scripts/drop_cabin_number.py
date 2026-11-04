"""刪除『船艙編號』欄位（已由『是否有船艙』保留資訊）。"""
import json
from datetime import datetime

import pandas as pd

WORKING = "dataset_streamlit_shell/workspace/working.csv"
LOG = "dataset_streamlit_shell/workspace/cleaning_log.jsonl"
DROP_COL = "船艙編號"

df = pd.read_csv(WORKING)
before_cols = df.shape[1]
rows_affected = int(df.shape[0])

df = df.drop(columns=[DROP_COL])
df.to_csv(WORKING, index=False, encoding="utf-8")

entry = {
    "created_at": datetime.now().replace(microsecond=0).isoformat(),
    "actor": "agent",
    "action": "drop_cabin_number",
    "columns": [DROP_COL],
    "rows": rows_affected,
    "note": "刪除船艙編號欄位，艙位資訊已由是否有船艙保留。",
}
with open(LOG, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("欄位數:", before_cols, "->", df.shape[1])
print(list(df.columns))
