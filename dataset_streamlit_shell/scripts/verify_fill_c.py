import pandas as pd

df = pd.read_csv('workspace/working.csv')
print("shape:", df.shape)
print("缺值總數:", int(df.isna().sum().sum()))
print("\n船票號碼 UNKNOWN 筆數:", int((df['船票號碼'] == 'UNKNOWN').sum()))
print("登船港口分布:")
print(df['登船港口'].value_counts())

lines = open('workspace/cleaning_log.jsonl', encoding='utf-8').read().strip().split('\n')
print("\nlog 最後一行:", lines[-1])
print("log 行數:", len(lines))
