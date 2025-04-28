import pandas as pd
import numpy as np

df = pd.read_csv('Employee_Salary.csv')

print('Information of Dataset:\n', df.info())
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n', df.tail().T)
print('Any 5 rows:\n', df.sample(5).T)

print('Total Number of Null Values in Dataset:', df.isna().sum())

df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Experience'].fillna(df['Experience'].mean(), inplace=True)

print('Total Number of Null Values in Dataset after filling missing values:', df.isna().sum())

print('Statistical information of Numerical Columns: \n', df.describe())

print('Groupwise Statistical Summary....')
print('\n-------------------------- Experience -----------------------\n')
print(df['Experience'].groupby(df['Gender']).describe())

print('\n-------------------------- Age -----------------------\n')
print(df['Age'].groupby(df['Gender']).describe())

print('\n-------------------------- Salary -----------------------\n')
print(df['Salary'].groupby(df['Gender']).describe())

df = pd.read_csv('iris.csv')
df = df.drop('Id', axis=1)
df.columns = ['SL', 'SW', 'PL', 'PW', 'Species']

print(df.head().T)

print('Statistical information of Numerical Columns: \n', df.describe())

print('Groupwise Statistical Summary....')

print('\n-------------------------- Sepal Length -----------------------\n')
print(df['SL'].groupby(df['Species']).describe())

print('\n-------------------------- Sepal Width -----------------------\n')
print(df['SW'].groupby(df['Species']).describe())

print('\n-------------------------- Petal Length -----------------------\n')
print(df['PL'].groupby(df['Species']).describe())

print('\n-------------------------- Petal Width -----------------------\n')
print(df['PW'].groupby(df['Species']).describe())
