import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer

def DetectOutlier(df, var):
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    high, low = Q3 + 1.5 * IQR, Q1 - 1.5 * IQR
    print(f"Highest allowed in {var}: {high}")
    print(f"Lowest allowed in {var}: {low}")
    count = df[(df[var] > high) | (df[var] < low)][var].count()
    print(f"Total outliers in {var}: {count}")
    df_outliers = df[((df[var] < low) | (df[var] > high))]
    print('Outliers:', len(df_outliers))
    print(df_outliers.T)
    df = df[((df[var] >= low) & (df[var] <= high))]
    return df

df = pd.read_csv('academic.csv')

print('Information of Dataset:\n', df.info())
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n', df.tail().T)
print('Any 5 rows:\n', df.sample(5).T)

print('Statistical information of Numerical Columns: \n', df.describe())

print('Total Number of Null Values in Dataset: \n', df.isna().sum())

df['gender'].fillna(df['gender'].mode()[0], inplace=True)
df['raisedhands'].fillna(df['raisedhands'].mean(), inplace=True)

print('Total Number of Null Values in Dataset after filling missing values: \n', df.isna().sum())

df['Relation'] = df['Relation'].astype('category')
df['Relation'] = df['Relation'].cat.codes

fig, axes = plt.subplots(2, 2)
fig.suptitle('Before removing Outliers')
sns.boxplot(data=df, x='raisedhands', ax=axes[0, 0])
sns.boxplot(data=df, x='VisITedResources', ax=axes[0, 1])
sns.boxplot(data=df, x='AnnouncementsView', ax=axes[1, 0])
sns.boxplot(data=df, x='Discussion', ax=axes[1, 1])
plt.show()

df = DetectOutlier(df, 'raisedhands')

fig, axes = plt.subplots(2, 2)
fig.suptitle('After removing Outliers')
sns.boxplot(data=df, x='raisedhands', ax=axes[0, 0])
sns.boxplot(data=df, x='VisITedResources', ax=axes[0, 1])
sns.boxplot(data=df, x='AnnouncementsView', ax=axes[1, 0])
sns.boxplot(data=df, x='Discussion', ax=axes[1, 1])
plt.show()

print('---------------- Data Skew Values before Yeo-John Transformation ----------------------')
print('raisedhands: ', df['raisedhands'].skew())
print('VisITedResources: ', df['VisITedResources'].skew())
print('AnnouncementsView: ', df['AnnouncementsView'].skew())
print('Discussion: ', df['Discussion'].skew())

fig, axes = plt.subplots(2, 2)
fig.suptitle('Handling Data Skewness')
sns.histplot(ax=axes[0, 0], data=df['AnnouncementsView'], kde=True)
sns.histplot(ax=axes[0, 1], data=df['Discussion'], kde=True)

yeojohnTr = PowerTransformer(standardize=True)
df['AnnouncementsView'] = yeojohnTr.fit_transform(df['AnnouncementsView'].values.reshape(-1, 1))
df['Discussion'] = yeojohnTr.fit_transform(df['Discussion'].values.reshape(-1, 1))

print('---------------- Data Skew Values after Yeo-John Transformation ----------------------')
print('AnnouncementsView: ', df['AnnouncementsView'].skew())
print('Discussion: ', df['Discussion'].skew())

sns.histplot(ax=axes[1, 0], data=df['AnnouncementsView'], kde=True)
sns.histplot(ax=axes[1, 1], data=df['Discussion'], kde=True)
plt.show()
