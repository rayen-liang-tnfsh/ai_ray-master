import json
from datetime import datetime

import pandas as pd

path = 'workspace/working.csv'
df = pd.read_csv(path)

mask = df['年齡'].isna()
fill_map = df.groupby('船票等級')['年齡'].mean()
print("各艙等平均年齡:")
print(fill_map.round(3))

df.loc[mask, '年齡'] = df.loc[mask, '船票等級'].map(fill_map)
df.to_csv(path, index=False, encoding='utf-8', lineterminator='\r\n')

print("\n填入筆數:", int(mask.sum()))
print("剩餘年齡缺值:", int(df['年齡'].isna().sum()))

record = {
    "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "actor": "agent",
    "action": "fill_missing_age_by_pclass",
    "columns": ["年齡"],
    "rows": int(mask.sum()),
    "note": "依船票等級分組（1等38.15、2等29.88、3等25.03歲）補齊全部 157 筆年齡缺值。",
}
with open('workspace/cleaning_log.jsonl', 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')
print("log 已寫入")
