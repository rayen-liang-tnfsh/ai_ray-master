import pandas as pd

df = pd.read_csv('workspace/working.csv')
print("shape:", df.shape)
print("年齡缺值:", int(df['年齡'].isna().sum()))
print("\n各艙等年齡統計:")
print(df.groupby('船票等級')['年齡'].agg(['count', 'mean', 'min', 'max']).round(2))

lines = open('workspace/cleaning_log.jsonl', encoding='utf-8').read().strip().split('\n')
print("\nlog 最後一行:", lines[-1])
print("log 行數:", len(lines))
