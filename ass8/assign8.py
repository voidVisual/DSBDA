# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# --------------------------------------------------------------------------------------
# Reading dataset
df = pd.read_csv('titanic.csv')
# --------------------------------------------------------------------------------------
# Display basic information
print('Information of Dataset:\n')
print(df.info())  # <- parentheses added
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns.tolist())
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):\n', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n', df.tail().T)
print('Any 5 rows:\n', df.sample(5).T)
# --------------------------------------------------------------------------------------
# Display Statistical information
print('Statistical information of Numerical Columns: \n', df.describe())
# --------------------------------------------------------------------------------------
# Display and fill the Null values
print('Total Number of Null Values in Dataset:\n', df.isna().sum())
df['Age'].fillna(df['Age'].median(), inplace=True)
print('Total Number of Null Values after filling:\n', df.isna().sum())
# --------------------------------------------------------------------------------------
# Single variable histogram - Categorical variables
fig, axis = plt.subplots(1, 3, figsize=(18,5))
sns.histplot(ax=axis[0], data=df, x='Sex', hue='Sex', multiple='dodge', shrink=0.8)
axis[0].set_title('Distribution of Sex')
sns.histplot(ax=axis[1], data=df, x='Pclass', hue='Pclass', multiple='dodge', shrink=0.8)
axis[1].set_title('Distribution of Pclass')
sns.histplot(ax=axis[2], data=df, x='Survived', hue='Survived', multiple='dodge', shrink=0.8)
axis[2].set_title('Distribution of Survival')
plt.tight_layout()
plt.show()

# Single variable histogram - Numerical variables
fig, axis = plt.subplots(1, 2, figsize=(14,5))
sns.histplot(ax=axis[0], data=df, x='Age', multiple='dodge', shrink=0.8, kde=True)
axis[0].set_title('Distribution of Age')
sns.histplot(ax=axis[1], data=df, x='Fare', multiple='dodge', shrink=0.8, kde=True)
axis[1].set_title('Distribution of Fare')
plt.tight_layout()
plt.show()

# Two variable histogram - Age and Fare with hue = Sex or Survived
fig, axis = plt.subplots(2, 2, figsize=(14,10))
sns.histplot(ax=axis[0,0], data=df, x='Age', hue='Sex', multiple='dodge', shrink=0.8, kde=True)
axis[0,0].set_title('Age Distribution by Sex')
sns.histplot(ax=axis[0,1], data=df, x='Fare', hue='Sex', multiple='dodge', shrink=0.8, kde=True)
axis[0,1].set_title('Fare Distribution by Sex')
sns.histplot(ax=axis[1,0], data=df, x='Age', hue='Survived', multiple='dodge', shrink=0.8, kde=True)
axis[1,0].set_title('Age Distribution by Survival')
sns.histplot(ax=axis[1,1], data=df, x='Fare', hue='Survived', multiple='dodge', shrink=0.8, kde=True)
axis[1,1].set_title('Fare Distribution by Survival')
plt.tight_layout()
plt.show()

# Two variable histogram - Categorical x hue
fig, axis = plt.subplots(2, 2, figsize=(14,10))
sns.histplot(ax=axis[0,0], data=df, x='Sex', hue='Survived', multiple='dodge', shrink=0.8)
axis[0,0].set_title('Sex vs Survival')
sns.histplot(ax=axis[0,1], data=df, x='Pclass', hue='Survived', multiple='dodge', shrink=0.8)
axis[0,1].set_title('Pclass vs Survival')
sns.histplot(ax=axis[1,0], data=df, x='Age', hue='Survived', multiple='dodge', shrink=0.8, kde=True)
axis[1,0].set_title('Age vs Survival')
sns.histplot(ax=axis[1,1], data=df, x='Fare', hue='Survived', multiple='dodge', shrink=0.8, kde=True)
axis[1,1].set_title('Fare vs Survival')
plt.tight_layout()
plt.show()
