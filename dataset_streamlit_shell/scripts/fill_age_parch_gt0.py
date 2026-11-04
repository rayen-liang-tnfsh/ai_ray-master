import json
from datetime import datetime

import pandas as pd

path = 'workspace/working.csv'
df = pd.read_csv(path)

mask_missing = df['年齡'].isna() & (df['同船父母與子女數'] > 0)
fill_value = df.loc[df['同船父母與子女數'] > 0, '年齡'].mean()

df.loc[mask_missing, '年齡'] = fill_value
df.to_csv(path, index=False, encoding='utf-8', lineterminator='\r\n')

print("填入值:", round(fill_value, 4))
print("填入筆數:", int(mask_missing.sum()))
print("剩餘年齡缺值:", int(df['年齡'].isna().sum()))

record = {
    "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "actor": "agent",
    "action": "fill_missing_age_by_parch",
    "columns": ["年齡"],
    "rows": int(mask_missing.sum()),
    "note": f"以同船父母與子女數>0 族群的平均年齡 {fill_value:.2f} 補齊 177 筆年齡缺值中符合條件者。",
}
with open('workspace/cleaning_log.jsonl', 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')
