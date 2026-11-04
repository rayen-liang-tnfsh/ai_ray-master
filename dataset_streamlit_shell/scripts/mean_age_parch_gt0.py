import pandas as pd

df = pd.read_csv('workspace/working.csv')

sub = df[df['同船父母與子女數'] > 0]
print("符合條件的筆數:", len(sub))
print("其中有年齡的筆數:", sub['年齡'].notna().sum())
print("年齡平均:", round(sub['年齡'].mean(), 2))
print("年齡中位數:", sub['年齡'].median())
print("年齡標準差:", round(sub['年齡'].std(), 2))
