import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# --------------------------------------------------------------------------------------
# Reading dataset
df = pd.read_csv('iris.csv')
df = df.drop('Id', axis=1)
df.columns = ('SL', 'SW', 'PL', 'PW', 'Species')

# Display basic information
print('Information of Dataset:\n')
print(df.info())  # <- Added parentheses
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns.tolist())
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):\n', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n', df.tail().T)
print('Any 5 rows:\n', df.sample(5).T)

# Display Statistical information
print('Statistical information of Numerical Columns: \n', df.describe())

# Display and fill the Null values
print('Total Number of Null Values in Dataset:\n', df.isna().sum())

# Boxplots for each numerical column
fig, axis = plt.subplots(2, 2, figsize=(12,8))
sns.boxplot(ax=axis[0,0], data=df, y='SL')
sns.boxplot(ax=axis[0,1], data=df, y='SW')
sns.boxplot(ax=axis[1,0], data=df, y='PL')
sns.boxplot(ax=axis[1,1], data=df, y='PW')
plt.tight_layout()
plt.show()

# Boxplots for each numerical column grouped by Species
fig, axis = plt.subplots(2, 2, figsize=(12,8))
sns.boxplot(ax=axis[0,0], data=df, y='SL', hue='Species')
sns.boxplot(ax=axis[0,1], data=df, y='SW', hue='Species')
sns.boxplot(ax=axis[1,0], data=df, y='PL', hue='Species')
sns.boxplot(ax=axis[1,1], data=df, y='PW', hue='Species')
plt.tight_layout()
plt.show()

# Histograms for each numerical column
fig, axis = plt.subplots(2, 2, figsize=(12,8))
sns.histplot(ax=axis[0,0], data=df, x='SL', multiple='dodge', shrink=0.8, kde=True)
sns.histplot(ax=axis[0,1], data=df, x='SW', multiple='dodge', shrink=0.8, kde=True)
sns.histplot(ax=axis[1,0], data=df, x='PL', multiple='dodge', shrink=0.8, kde=True)
sns.histplot(ax=axis[1,1], data=df, x='PW', multiple='dodge', shrink=0.8, kde=True)
plt.tight_layout()
plt.show()

# Histograms for each numerical column grouped by Species
fig, axis = plt.subplots(2, 2, figsize=(12,8))
sns.histplot(ax=axis[0,0], data=df, x='SL', hue='Species', element='poly', shrink=0.8, kde=True)
sns.histplot(ax=axis[0,1], data=df, x='SW', hue='Species', element='poly', shrink=0.8, kde=True)
sns.histplot(ax=axis[1,0], data=df, x='PL', hue='Species', element='poly', shrink=0.8, kde=True)
sns.histplot(ax=axis[1,1], data=df, x='PW', hue='Species', element='poly', shrink=0.8, kde=True)
plt.tight_layout()
plt.show()
