import pandas as pd

df = pd.read_csv('workspace/working.csv')

print("船票等級缺值:", int(df['船票等級'].isna().sum()))
print("\n各艙等平均年齡（僅用有年齡者計算）：")
print(df.groupby('船票等級')['年齡'].agg(['count', 'mean']).round(3))

print("\n各艙等的年齡缺值筆數：")
print(df[df['年齡'].isna()].groupby('船票等級').size())
