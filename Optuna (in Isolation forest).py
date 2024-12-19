import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import optuna

df = pd.read_csv('encoded_healthcare_providers.csv')


cols = ['Number of Services', 
        'Number of Medicare Beneficiaries', 
        'Number of Distinct Medicare Beneficiary/Per Day Services', 
        'Average Medicare Allowed Amount', 
        'Average Submitted Charge Amount', 
        'Average Medicare Payment Amount', 
        'Average Medicare Standardized Amount']

for col in cols:
    df[col] = df[col].replace(',', '', regex=True)  
    df[col] = pd.to_numeric(df[col], errors='coerce')  

def objective(trial):
    contamination = trial.suggest_float('contamination', 0.01, 0.1)
    isolation_forest = IsolationForest(contamination=contamination, random_state=42)

    df['Anomaly_IsolationForest'] = isolation_forest.fit_predict(df[cols])
    
    n_anomalies = np.sum(df['Anomaly_IsolationForest'] == -1)
    
    return n_anomalies

study = optuna.create_study(direction='minimize') 
study.optimize(objective, n_trials=100)  


print("Best parameters: ", study.best_params)


best_contamination = study.best_params['contamination']
isolation_forest = IsolationForest(contamination=best_contamination, random_state=42)
df['Anomaly_IsolationForest'] = isolation_forest.fit_predict(df[cols])

pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[cols])

df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
df_pca['anomaly'] = df['Anomaly_IsolationForest']


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='anomaly',
                palette={-1: 'red', 1: 'blue'}, alpha=0.6)

plt.title('PCA Projection of Isolation Forest Anomalies')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Anomaly', loc='upper right', labels=['Normal', 'Anomaly'])
plt.show()


