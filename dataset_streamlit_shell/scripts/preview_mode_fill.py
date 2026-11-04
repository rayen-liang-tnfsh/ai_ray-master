import pandas as pd

df = pd.read_csv('workspace/working.csv')

for c in ['登船港口', '船票號碼', '票價']:
    miss = int(df[c].isna().sum())
    mode_vals = df[c].mode(dropna=True)
    print(f"=== {c} ===")
    print(f"  缺值筆數: {miss}")
    print(f"  dtype: {df[c].dtype}")
    print(f"  眾數候選: {list(mode_vals[:3])}")
    print(f"  眾數出現次數: {int((df[c] == mode_vals.iloc[0]).sum())}")
    print(f"  不重複值數: {df[c].nunique()}")
    print()
