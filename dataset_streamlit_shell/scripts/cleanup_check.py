import pandas as pd, os
base = 'dataset_streamlit_shell/workspace'
pas = pd.read_csv(base + '/integration/passengers.csv')
voy = pd.read_csv(base + '/integration/voyage.csv')
m = pas.merge(voy, on='PassengerId', how='left')
print('matched Fare rows:', int(m['Fare'].notna().sum()), '/', len(m))
print('voyage id range:', voy.PassengerId.min(), voy.PassengerId.max())
print('passenger id range:', pas.PassengerId.min(), pas.PassengerId.max())
for f in ['passengers.csv', 'voyage.csv']:
    os.remove(base + '/integration/' + f)
print('integration left:', os.listdir(base + '/integration'))
