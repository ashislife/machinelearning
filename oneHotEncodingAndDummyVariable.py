import pandas as pd


dataset=pd.read_csv("sample_loan_data.csv")
# print(dataset.head(5))

# print()
# # check the NaN values
# print(dataset.isnull().sum())

# hot encoding use karne se pahle pure catagerical data ko fill karna hoga

print()
# fill the data (avoid the inplace error)
dataset.loc[:, "Gender"].fillna(dataset["Gender"].mode()[0], inplace=True)
dataset.loc[:, "Married"].fillna(dataset["Married"].mode()[0], inplace=True)
print(dataset.isnull().sum())

print()
en_data=dataset[["Gender","Married"]]
print(en_data)


# encoding perform (yes/no)
print()
print(pd.get_dummies(en_data))

# hme numerical output chahiye esliye skitlean lib use karenge'
print()

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()

# print in form of array
arr=ohe.fit_transform(en_data).toarray()
# print(arr)

# print in form of Dataset
print(pd.DataFrame(arr,columns=["Gender_Female",  "Gender_Male",  "Married_No",  "Married_Yes"]))





