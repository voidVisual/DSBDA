
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading dataset
df = pd.read_csv('titanic.csv')

# Display basic information
print('Information of Dataset:\n')
print(df.info())  # <- added parentheses
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
df['Age'].fillna(df['Age'].median(), inplace=True)  # Filling 'Age' null values
print('Total Number of Null Values after filling:\n', df.isna().sum())



# One variable - Boxplot for 'Age' and 'Fare'
fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.boxplot(data=df, y='Age', ax=axes[0])
axes[0].set_title('Boxplot of Age')
sns.boxplot(data=df, y='Fare', ax=axes[1])
axes[1].set_title('Boxplot of Fare')
plt.tight_layout()
plt.show()

# Two variables - Boxplot 'Age' with different categories
fig, axes = plt.subplots(1, 3, figsize=(18,5), sharey=True)
sns.boxplot(data=df, x='Sex', y='Age', hue='Sex', ax=axes[0])
axes[0].set_title('Age Distribution by Sex')
sns.boxplot(data=df, x='Pclass', y='Age', hue='Pclass', ax=axes[1])
axes[1].set_title('Age Distribution by Passenger Class')
sns.boxplot(data=df, x='Survived', y='Age', hue='Survived', ax=axes[2])
axes[2].set_title('Age Distribution by Survival')
plt.tight_layout()
plt.show()

# Two variables - Boxplot 'Fare' with log scale
fig, axes = plt.subplots(1, 3, figsize=(18,5), sharey=True)
sns.boxplot(data=df, x='Sex', y='Fare', hue='Sex', ax=axes[0])
axes[0].set(yscale='log')
axes[0].set_title('Fare Distribution by Sex (Log Scale)')
sns.boxplot(data=df, x='Pclass', y='Fare', hue='Pclass', ax=axes[1])
axes[1].set(yscale='log')
axes[1].set_title('Fare Distribution by Passenger Class (Log Scale)')
sns.boxplot(data=df, x='Survived', y='Fare', hue='Survived', ax=axes[2])
axes[2].set(yscale='log')
axes[2].set_title('Fare Distribution by Survival (Log Scale)')
plt.tight_layout()
plt.show()

# Three variables - Boxplot for 'Age' based on 'Sex' and 'Survived'
fig, axes = plt.subplots(1, 2, figsize=(14,6), sharey=True)
sns.boxplot(data=df, x='Sex', y='Age', hue='Survived', ax=axes[0])
axes[0].set_title('Age vs Sex vs Survival')
sns.boxplot(data=df, x='Pclass', y='Age', hue='Survived', ax=axes[1])
axes[1].set_title('Age vs Pclass vs Survival')
plt.tight_layout()
plt.show()

# Three variables - Boxplot for 'Fare' based on 'Sex' and 'Survived' (with log scale)
fig, axes = plt.subplots(1, 2, figsize=(14,6), sharey=True)
sns.boxplot(data=df, x='Sex', y='Fare', hue='Survived', ax=axes[0])
axes[0].set(yscale='log')
axes[0].set_title('Fare vs Sex vs Survival (Log Scale)')
sns.boxplot(data=df, x='Pclass', y='Fare', hue='Survived', ax=axes[1])
axes[1].set(yscale='log')
axes[1].set_title('Fare vs Pclass vs Survival (Log Scale)')
plt.tight_layout()
plt.show()