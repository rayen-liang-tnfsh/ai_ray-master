import pandas as pd

df = pd.read_csv("dataset_streamlit_shell/workspace/working.csv")
print("shape:", df.shape)
print("Name 空值:", int(df["Name"].isna().sum()))

s = df["Name"].astype("string")
dup_mask = s.duplicated(keep=False)
dup = df[dup_mask].sort_values("Name")

print("重複涉及的筆數:", int(dup_mask.sum()))
print("重複的不同 Name 個數:", int(s[dup_mask].nunique()))
print("--- 重複明細 ---")
print(dup[["PassengerId", "Name", "Sex", "Age", "Ticket", "Fare", "Embarked"]].to_string(index=False))
print("--- 次數統計 ---")
print(s[dup_mask].value_counts().to_string())
