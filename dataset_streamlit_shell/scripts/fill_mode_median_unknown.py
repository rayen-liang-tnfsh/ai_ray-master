import json
from datetime import datetime

import pandas as pd

path = 'workspace/working.csv'
df = pd.read_csv(path)

# 1. 登船港口：眾數填補
embark_mode = df['登船港口'].mode(dropna=True).iloc[0]
m_emb = df['登船港口'].isna()
df.loc[m_emb, '登船港口'] = embark_mode

# 2. 票價：中位數填補
fare_median = df['票價'].median()
m_fare = df['票價'].isna()
df.loc[m_fare, '票價'] = fare_median

# 3. 船票號碼：填 UNKNOWN
m_ticket = df['船票號碼'].isna()
df.loc[m_ticket, '船票號碼'] = 'UNKNOWN'

df.to_csv(path, index=False, encoding='utf-8', lineterminator='\r\n')

print("登船港口眾數:", embark_mode, "| 填入:", int(m_emb.sum()))
print("票價中位數:", fare_median, "| 填入:", int(m_fare.sum()))
print("船票號碼 UNKNOWN | 填入:", int(m_ticket.sum()))
print("剩餘缺值:", {c: int(df[c].isna().sum()) for c in df.columns})

record = {
    "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "actor": "agent",
    "action": "fill_missing_by_mode_median_unknown",
    "columns": ["登船港口", "票價", "船票號碼"],
    "rows": int(m_emb.sum() + m_fare.sum() + m_ticket.sum()),
    "note": "登船港口以眾數 S 補 6 筆、票價以中位數 %.2f 補 4 筆、船票號碼缺值以 UNKNOWN 標記 4 筆。" % fare_median,
}
with open('workspace/cleaning_log.jsonl', 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')
print("log 已寫入")
