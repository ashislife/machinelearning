import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv("sample_loan_data.csv")
print(dataset)

print()


# print(dataset.dropna(inplace=True))
# print(dataset.to_csv("sample_loan_data.csv", index=False))


print()
print(dataset.isnull().sum())
print()


# fill the missing content
# print(dataset.fillna(10))


print()

# check the data info
print(dataset.info())
print()

# backward filling
# print(dataset.fillna(method="bfill"))

print()

# forward filling
# print(dataset.fillna(method="ffill"))

print()

# fill colomnwise
# print(dataset.fillna(method="ffill",axis=1))


print()

# use mod data =ye sabse jyada repeat data ko NaN ke jagah fill karta haiii
print(dataset["Married"].mode()[0])

print()

# Nan value ko sabse jyada repeated value se replace kr dega
print(dataset["Married"].fillna(dataset["Married"].mode()[0]))

print()

# jitne v object type data hai usko dekhne k eliye
print(dataset.select_dtypes(include="object"))

print()

# whole data ke upper jo sabse jyada repeated value hai usse fill karna
for i in dataset.select_dtypes(include="object").columns:
    dataset[i] =dataset[i].fillna(dataset[i].mode()[0])
print(dataset.isnull().sum())


