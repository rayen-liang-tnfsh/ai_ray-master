import pandas as pd

p = 'dataset_streamlit_shell/workspace/integration/voyage.csv'
df = pd.read_csv(p)
df = df.rename(columns={'passenger_id': 'PassengerId'})
df.to_csv(p, index=False)
print('columns:', list(df.columns), 'rows:', len(df))
