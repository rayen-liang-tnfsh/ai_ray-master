import pandas as pd

base = 'dataset_streamlit_shell/workspace'
pas = pd.read_csv(f'{base}/integration/passengers.csv')
voy = pd.read_csv(f'{base}/integration/voyage.csv')

print('passengers:', pas.shape, list(pas.columns))
print('voyage    :', voy.shape, list(voy.columns))

merged = pas.merge(voy, on='PassengerId', how='left')
print('merged    :', merged.shape, list(merged.columns))

merged.to_csv(f'{base}/original.csv', index=False)
merged.to_csv(f'{base}/working.csv', index=False)
print('written original.csv & working.csv')
