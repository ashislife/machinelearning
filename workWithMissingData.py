import pandas as pd

dataset=pd.read_csv(r"D:\ML practical\loan_dataset.csv")

# print top 5 datas
# print(dataset.head(5))


# count null values
print(dataset.isnull().sum())

print()

# count total null values
print("Total null values =",dataset.isnull().sum().sum())

print()

# calculate null value in percentage
print(dataset.isnull().sum()/dataset.shape[0]*100)

print(dataset.shape)
