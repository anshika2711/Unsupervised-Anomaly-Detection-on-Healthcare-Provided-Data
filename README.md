# Unsupervised-Anomaly-Detection-on-HealthCare-Providers-Dataset
---
## Overview  
This project focuses on identifying anomalies in healthcare provider data using various machine learning techniques. The dataset contains information such as provider identifiers, services rendered, and payment amounts. The primary goal is to detect unusual patterns or outliers that deviate significantly from the norm.

---

## Workflow

### 1. **Data Preprocessing**
- Handled missing values and removed duplicates.
- Adjusted column data types for consistency.
- Visualized the data to understand distributions and correlations.
- Encoded categorical columns using:
  - **One-Hot Encoding**
  - **Frequency Encoding**

### 2. **Outlier Removal**
- Experimented with:
  - **Z-score Method**
  - **IQR Method**
- Finalized the **IQR Method** for outlier removal.

### 3. **Anomaly Detection**
- Applied multiple anomaly detection techniques:
  - **Isolation Forest** (best-performing model)
  - **One-Class SVM**
  - **Local Outlier Factor (LOF)**
- Hyperparameter Tuning:
  - **Grid Search with GridSearchCV**
  - **Randomized Search**
  - **Optimization using Optuna**
- Evaluated models using:
  - **Silhouette Score**
  - **Calinski-Harabasz Score**
- Compared results using a confusion matrix.

### 4. **Autoencoder Implementation**
- Designed a **vanilla autoencoder** for anomaly detection.
- Tuned parameters and visualized reconstruction error.
- Compared autoencoder results with the Isolation Forest to finalize the best-performing model.

---

## Dataset Details
- **Size**: 100,000 rows and 27 columns.
- **Numeric Columns**:
  - Number of Services
  - Number of Medicare Beneficiaries
  - Average Medicare Allowed Amount
  - Average Submitted Charge Amount
  - Average Medicare Payment Amount
  - Number of Distinct Medicare Beneficiary/Per Day Services
  - Average Medicare Standardized Amount
- **Categorical Columns**: Encoded using various encoding techniques.
- **Dataset Link**: [Healthcare Providers Data](https://www.kaggle.com/datasets/tamilsel/healthcare-providers-data)

---

## Key Results
- The **Isolation Forest** technique provided the best results in anomaly detection.
- Autoencoder reconstruction error analysis helped validate the model's performance.

---
