# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE


data = pd.read_csv('Final_Healthcare_Providers_Dataset.csv')


X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values   

print("Initial shapes:")
print("X shape:", X.shape)
print("y shape:", y.shape)


threshold = 0.5  
y_binary = (y > threshold).astype(int)  


X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)


print("Shapes after splitting:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Scale the features to improve convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#class imbalance 
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


print("Shapes after SMOTE:")
print("X_train_resampled shape:", X_train_resampled.shape)
print("y_train_resampled shape:", y_train_resampled.shape)


input_dim = X_train.shape[1]  

# Encoder
input_layer = Input(shape=(input_dim,))
encoder = Dense(128, activation='relu')(input_layer)
encoder = Dense(64, activation='relu')(encoder)
encoder = Dense(32, activation='relu')(encoder)

# Decoder
decoder = Dense(64, activation='relu')(encoder)
decoder = Dense(128, activation='relu')(decoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

# Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Training the Autoencoder with Early Stopping
early_stopping_monitor = EarlyStopping(monitor='loss', patience=5)
autoencoder.fit(X_train_resampled, X_train_resampled,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping_monitor])


feature_extractor = Model(inputs=input_layer, outputs=encoder)
X_train_encoded = feature_extractor.predict(X_train_resampled)
X_test_encoded = feature_extractor.predict(X_test)

classifier = Sequential()
classifier.add(Dense(32, activation='relu', input_dim=X_train_encoded.shape[1]))
classifier.add(Dense(16, activation='relu'))
classifier.add(Dense(1, activation='sigmoid')) 

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


classifier.fit(X_train_encoded, y_train_resampled,
               epochs=100,
               batch_size=32,
               validation_split=0.2,
               callbacks=[early_stopping_monitor])


y_pred_prob = classifier.predict(X_test_encoded)
y_pred = (y_pred_prob > 0.5).astype(int)


reconstructed_X_test = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.square(X_test - reconstructed_X_test), axis=1)


plt.figure(figsize=(10, 6))
sns.histplot(reconstruction_error[y_test == 0], bins=50, color='blue', label='Normal', kde=True, alpha=0.5)
sns.histplot(reconstruction_error[y_test == 1], bins=50, color='red', label='Anomaly', kde=True, alpha=0.5)
plt.title('Reconstruction Error Distribution')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()


report = classification_report(y_test, y_pred, output_dict=True)
print("Precision:", report['1']['precision'])
print("Recall:", report['1']['recall'])
print("F1 Score:", report['1']['f1-score'])
print("Accuracy:", report['accuracy'])

conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


