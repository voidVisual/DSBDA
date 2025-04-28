import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def RemoveOutlier(df, var):
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    high, low = Q3 + 1.5*IQR, Q1 - 1.5*IQR
    print("Highest allowed in variable:", var, high)
    print("Lowest allowed in variable:", var, low)
    count = df[(df[var] > high) | (df[var] < low)][var].count()
    print('Total outliers in', var, ':', count)
    df = df[(df[var] >= low) & (df[var] <= high)]
    return df

def BuildModel(X, Y):
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import confusion_matrix, classification_report

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.25, random_state=0)
    model = GaussianNB()
    model = model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    cm = confusion_matrix(ytest, ypred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.show()
    print(classification_report(ytest, ypred))

df = pd.read_csv('iris.csv')
df = df.drop('Id', axis=1)
df.columns = ('SL', 'SW', 'PL', 'PW', 'Species')
print('Information of Dataset:\n', df.info())
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns.tolist())
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):\n', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n', df.tail().T)
print('Any 5 rows:\n', df.sample(5).T)

print('Statistical information of Numerical Columns: \n', df.describe())

print('Total Number of Null Values in Dataset:\n', df.isna().sum())
df['SL'].fillna(df['SL'].mean(), inplace=True)
df['PW'].fillna(df['PW'].mean(), inplace=True)
df.isna().sum()

df['Species'] = df['Species'].astype('category')
df['Species'] = df['Species'].cat.codes

sns.heatmap(df.corr(), annot=True)
plt.title('Correlation Matrix')
plt.show()

X = df[['SL', 'SW', 'PL', 'PW']]
Y = df['Species']
BuildModel(X, Y)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(data=df, y='SL', ax=axes[0, 0])
sns.boxplot(data=df, y='SW', ax=axes[0, 1])
sns.boxplot(data=df, y='PL', ax=axes[1, 0])
sns.boxplot(data=df, y='PW', ax=axes[1, 1])
plt.tight_layout()
plt.show()

df = RemoveOutlier(df, 'SW')   
df.isna().sum()

X = df[['SL', 'SW', 'PL', 'PW']]
Y = df['Species']
BuildModel(X, Y)
