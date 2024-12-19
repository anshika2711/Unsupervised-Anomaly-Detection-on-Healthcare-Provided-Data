# %%
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import KNNImputer
warnings.filterwarnings("ignore")


# %%
df =pd.read_csv("Healthcare Providers.csv", skipinitialspace=True)
print("Missing values:\n",df.isnull().sum())
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)


# %%
df['Number of Services'] = df['Number of Services'].astype(str).str.replace(',', '')
df['Number of Medicare Beneficiaries'] = df['Number of Medicare Beneficiaries'].astype(str).str.replace(',', '')

# %%
df['Number of Services'] = df['Number of Services'].astype(float)
df['Number of Medicare Beneficiaries'] = df['Number of Medicare Beneficiaries'].astype(float)

# %%
df['Number of Services'] = df['Number of Services'].round(0).astype(int)
df['Number of Medicare Beneficiaries'] = df['Number of Medicare Beneficiaries'].round(0).astype(int)
print(df[['Number of Services', 'Number of Medicare Beneficiaries']].head())

# %%
df['National Provider Identifier'] = df['National Provider Identifier'].astype(str)
df['Number of Services'] = df['Number of Services'].astype(int)
df['Number of Medicare Beneficiaries'] =df['Number of Medicare Beneficiaries'].astype(int)
# %%
print(df.info())
# %%
df.to_csv('Healthcare Providers.csv', index=False)
# %%
df['Average Submitted Charge Amount'] = pd.to_numeric(df['Average Submitted Charge Amount'], errors='coerce')
df = df.dropna(subset=['Average Submitted Charge Amount'])









