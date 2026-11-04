import pandas as pd

df = pd.read_csv('workspace/working.csv')
print("shape:", df.shape)
cols = ['登船港口_C', '登船港口_Q', '登船港口_S']
print("每筆三欄加總分布:")
print(df[cols].sum(axis=1).value_counts())
print("\n各欄 1 的筆數:", {c: int(df[c].sum()) for c in cols})
print("\n缺值總數:", int(df.isna().sum().sum()))
