import pandas as pd

df = pd.read_csv("netflix_titles.csv")

print("Dataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDescriptive Statistics:")
print(df.describe(include='all'))

print("\nFirst 5 rows:")
print(df.head())

