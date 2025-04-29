import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def RemoveOutlier(df, var):
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    high, low = Q3 + 1.5 * IQR, Q1 - 1.5 * IQR
    print("Highest allowed in variable:", var, high)
    print("Lowest allowed in variable:", var, low)
    count = df[(df[var] > high) | (df[var] < low)][var].count()
    print('Total outliers in:', var, ':', count)
    df = df[((df[var] >= low) & (df[var] <= high))]
    return df

def BuildModel(X, Y):
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.25, random_state=13)
    model = LogisticRegression(solver='lbfgs')
    model = model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    cm = confusion_matrix(ytest, ypred)
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()
    print(classification_report(ytest, ypred))

df = pd.read_csv('purchase.csv')
print('Information of Dataset:\n', df.info())
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n', df.tail().T)
print('Any 5 rows:\n', df.sample(5).T)

df = df.drop('User ID', axis=1)
df.columns = ['Gender', 'Age', 'Salary', 'Purchased']

print('Statistical information of Numerical Columns: \n', df.describe())
print('Total Number of Null Values in Dataset:', df.isna().sum())

df['Gender'] = df['Gender'].astype('category')
df['Gender'] = df['Gender'].cat.codes

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()

X = df[['Age', 'Salary']]
Y = df['Purchased']
BuildModel(X, Y)

fig, axes = plt.subplots(1, 2)
sns.boxplot(data=df, x='Age', ax=axes[0])
sns.boxplot(data=df, x='Salary', ax=axes[1])
fig.tight_layout()
plt.show()

df = RemoveOutlier(df, 'Age')
df = RemoveOutlier(df, 'Salary')

X = df[['Age', 'Salary']]
Y = df['Purchased']
BuildModel(X, Y)
