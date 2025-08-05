# give the wrong output

import pandas as pd
df=pd.read_csv("dummiesDataset.csv")
print(df)
print()

# use LabelEncoder
from sklearn.preprocessing import LabelEncoder

# create an object for LabelEncoder
le=LabelEncoder()

dfle=df

# categorical data replace with numerical data
dfle.Town=le.fit_transform(dfle.Town)
print("town transform")
print(dfle)

print()
# X-->independent variable
print("X dataset")
X=dfle[['Town','Area']].values
print(X)

print()
print("Y dataset")
# Y-dependent variable
Y=dfle.Price
print(Y)

print()
# use onehotencoding
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()

arr=ohe.fit_transform(X).toarray()
print(arr)
