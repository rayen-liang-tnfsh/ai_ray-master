import json

import pandas as pd

df = pd.read_csv('workspace/working.csv')
print("shape:", df.shape)
print("剩餘年齡缺值:", int(df['年齡'].isna().sum()))
print("parch>0 且年齡缺值:", int(df.loc[df['同船父母與子女數'] > 0, '年齡'].isna().sum()))

lines = open('workspace/cleaning_log.jsonl', encoding='utf-8').read().strip().split('\n')
print("log 最後一行:", lines[-1])
print("log 行數:", len(lines))
