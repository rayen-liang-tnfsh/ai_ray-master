import pandas as pd

df = pd.read_csv("dataset_streamlit_shell/workspace/working.csv")
print(df["Embarked"].value_counts(dropna=False).to_string())
print("--- repr ---")
print(df["Embarked"].map(repr).value_counts(dropna=False).to_string())
