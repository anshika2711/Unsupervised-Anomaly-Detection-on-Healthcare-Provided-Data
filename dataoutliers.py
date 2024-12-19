import pandas as pd
import numpy as np
from scipy.stats import zscore
file_path = 'Healthcare Providers.csv'
data = pd.read_csv(file_path)

# remove outliers using the Z-score method
def remove_outliers_zscore(data, threshold=3):
    
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    z_scores = data[numerical_cols].apply(zscore)
    data_filtered = data[(np.abs(z_scores) < threshold).all(axis=1)]
    return data_filtered

# Remove outliers from the dataset using the Z-score method
data_cleaned = remove_outliers_zscore(data, threshold=3)
data_cleaned.to_csv('Cleaned_Healthcare_Providers.csv', index=False)

# %% Visualizing Outliers Removed with Box Plots
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Cleaned_Healthcare_Providers.csv')

# Identify numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Create box plots for each numerical column to visualize outliers
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col} (Outliers Removed)')
    plt.ylabel(col)
    plt.show()

# %% Calculating and Removing Outliers Based on Standard Deviation
upper_limit = df['Number of Services'].mean() + 3 * df['Number of Services'].std()
lower_limit = df['Number of Services'].mean() - 3 * df['Number of Services'].std()

print('Upper limit:', upper_limit)
print('Lower limit:', lower_limit)

# Filter out outliers
new_df = df[(df['Number of Services'] < upper_limit) & (df['Number of Services'] > lower_limit)]

# Display the effect of outlier removal
print('Before removing outliers:', len(df))
print('After removing outliers:', len(new_df))
print('Number of outliers removed:', len(df) - len(new_df))

# %% Visualizing the 'Number of Services' after outlier removal
sns.boxplot(new_df['Number of Services'])
plt.show()








