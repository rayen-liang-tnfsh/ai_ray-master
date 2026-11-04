import json
from datetime import datetime

import pandas as pd

path = 'workspace/working.csv'
df = pd.read_csv(path)

dummies = pd.get_dummies(df['登船港口'], prefix='登船港口', dtype=int)
print("產生的欄位:", list(dummies.columns))

insert_at = list(df.columns).index('登船港口') + 1
df = pd.concat([df.iloc[:, :insert_at], dummies, df.iloc[:, insert_at:]], axis=1)
df = df.drop(columns=['登船港口'])

df.to_csv(path, index=False, encoding='utf-8', lineterminator='\r\n')

print("\n欄位數:", df.shape[1], "筆數:", len(df))
print("欄位:", list(df.columns))
print(dummies.sum().to_dict())

record = {
    "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "actor": "agent",
    "action": "one_hot_encode_embarked",
    "columns": ["登船港口", "登船港口_S", "登船港口_C", "登船港口_Q"],
    "rows": len(df),
    "note": "將登船港口欄位 one-hot 編碼為 S／C／Q 三欄（0/1），並移除原始文字欄位。",
}
with open('workspace/cleaning_log.jsonl', 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')
print("log 已寫入")
