import pandas as pd

df = pd.read_csv('workspace/working.csv')
print("性別 dtype:", df['性別'].dtype)
print("性別值分布:")
print(df['性別'].value_counts(dropna=False))
print("\n目前欄位:", list(df.columns))
