import pandas as pd

df = pd.read_csv("dataset_streamlit_shell/workspace/working.csv")
print("shape:", df.shape)
print("--- Sex ---")
print(df["Sex"].value_counts(dropna=False).to_string())
print("--- Embarked ---")
print(df["Embarked"].value_counts(dropna=False).to_string())
