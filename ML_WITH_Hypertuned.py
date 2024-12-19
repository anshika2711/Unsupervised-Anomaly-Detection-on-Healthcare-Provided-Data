import pandas as pd
df = pd.read_csv('encoded_healthcare_providers.csv')
df.replace({True: 1, False: 0}, inplace=True)
output_file_path = 'encoded_healthcare_providers.csv' 
df.to_csv(output_file_path, index=False)
print(f"File saved to {output_file_path}")

#One-Class SVM
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data = pd.read_csv('encoded_healthcare_providers.csv')

selected_columns = [    'Number of Services', 
    'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services', 
    'Average Medicare Allowed Amount', 
    'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 
    'Average Medicare Standardized Amount'
] 
data_selected = data[selected_columns]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected)

# Apply One-Class SVM to get anomaly labels
oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
oc_svm.fit(data_scaled)
labels = oc_svm.predict(data_scaled)

# Convert labels to binary for classification (1 = anomaly, 0 = normal)
labels = np.where(labels == -1, 1, 0)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.3, random_state=42)

# Apply an SVM classifier for binary classification
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train, y_train)

# Predict on test data
predictions = classifier.predict(X_test)

# Perform PCA for 2D visualization
pca = PCA(n_components=2)
data_2d = pca.fit_transform(X_test)

# Separate normal and anomalous data based on predictions
normal_data_2d = data_2d[predictions == 0]
anomalies_data_2d = data_2d[predictions == 1]

# Plot the 2D classification result
plt.figure(figsize=(10, 6))
plt.scatter(normal_data_2d[:, 0], normal_data_2d[:, 1], color='blue', label='Normal', s=20)
plt.scatter(anomalies_data_2d[:, 0], anomalies_data_2d[:, 1], color='red', label='Anomalies', s=20)
plt.title('Classification of Anomalies (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
oc_svm.fit(data_scaled)
predictions = oc_svm.predict(data_scaled)

anomaly_count = np.sum(predictions == -1)
print(f"Number of anomalies detected by One-Class SVM: {anomaly_count}")



#KNN
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('encoded_healthcare_providers.csv')
numerical_features = [
    'Number of Services', 
    'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services', 
    'Average Medicare Allowed Amount', 
    'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 
    'Average Medicare Standardized Amount'
]
data_numerical = data[numerical_features]

# Standardizing the numerical data for KNN
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numerical)

# Applying KNN for anomaly detection
knn = NearestNeighbors(n_neighbors=5)
knn.fit(data_scaled)

# Calculating anomaly scores using the distances to the nearest neighbors
distances, _ = knn.kneighbors(data_scaled)
anomaly_scores = np.mean(distances, axis=1)

# Defining anomalies as points with scores above a threshold (mean + 2 std deviations)
threshold = anomaly_scores.mean() + 2 * anomaly_scores.std()
anomalies = anomaly_scores > threshold

# Adding anomaly column and scores to the dataframe for visualization
data['Anomaly'] = anomalies
data['Anomaly Score'] = anomaly_scores
anomaly_count = anomalies.sum()
print(f"Total Anomalies Detected: {anomaly_count}")

# Scatter plot to show anomalies in two of the numerical feature dimensions
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average Medicare Allowed Amount', y='Average Submitted Charge Amount', 
                hue='Anomaly', data=data, palette={False: 'blue', True: 'red'}, alpha=0.6)
plt.title("KNN Anomalies in Healthcare Providers Dataset")
plt.xlabel("Average Medicare Allowed Amount")
plt.ylabel("Average Submitted Charge Amount")
plt.legend(title="Anomaly", labels=["Normal", "Anomalous"])
plt.show()


#Isolation_forest_with_optimization(hypertuning)
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('encoded_healthcare_providers.csv')
cols = ['Number of Services', 'Number of Medicare Beneficiaries', 
        'Number of Distinct Medicare Beneficiary/Per Day Services', 
        'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
        'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']
for col in cols:
    df[col] = df[col].replace(',', '', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=cols)
df[cols].isnull().sum()

#=================loop for optimization===============

best_score = float('-inf')
best_params = {}  

for n_estimators in [50, 100, 150]: 
    for max_samples in [0.4,'auto']: 
        for contamination in [0.05, 0.1, 0.15]: 
            model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, random_state=42)
            model.fit(df[cols])  
            score = model.decision_function(df[cols]).mean()  
            
            
            if score > best_score:
                best_score = score
                best_params = {'n_estimators': n_estimators, 'max_samples': max_samples, 'contamination': contamination}

print("Best Parameters:", best_params)  
print("Best Score:", best_score)  


final_model = IsolationForest(
    n_estimators=best_params['n_estimators'], 
    max_samples=best_params['max_samples'], 
    contamination=best_params['contamination'],
    random_state=42
)
df['Anomaly'] = final_model.fit_predict(df[cols])


# ========== PCA for Visualization =============
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[cols])
df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
df_pca['anomaly'] = df['Anomaly']
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='anomaly', palette={-1: 'red', 1: 'blue'}, alpha=0.6)
plt.title('PCA Projection of Isolation Forest Anomalies')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Anomaly', loc='upper right', labels=['Normal', 'Anomaly'])
plt.show()

from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.05, random_state=42)  
iso_forest.fit(df[cols])
predictions = iso_forest.predict(df[cols])

anomaly_count = np.sum(predictions == -1)
print(f"Number of anomalies detected by Isolation Forest: {anomaly_count}")



