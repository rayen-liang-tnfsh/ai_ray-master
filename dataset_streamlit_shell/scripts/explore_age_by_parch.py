import pandas as pd

df = pd.read_csv('workspace/working.csv')
print("shape:", df.shape)
print("年齡缺值:", int(df['年齡'].isna().sum()))

g = df.groupby('同船父母與子女數')['年齡'].agg(['count', 'median', 'mean', 'min', 'max'])
print("\n--- 依 同船父母與子女數 分組的年齡統計 ---")
print(g)

print("\n相關係數 (Parch vs Age):", round(df[['同船父母與子女數', '年齡']].corr().iloc[0, 1], 3))

print("\n--- 各組生還率 ---")
print(df.groupby('同船父母與子女數')['是否生還'].agg(['count', 'mean']))
