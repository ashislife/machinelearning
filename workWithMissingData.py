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

# calculate null value in percentage of each colom
print(dataset.isnull().sum()/dataset.shape[0]*100)

print()

# calculate how much no of null value present in overall dataset
print("total null value in %  ",dataset.isnull().sum().sum()/(dataset.shape[0]*dataset.shape[1])*100)

print()


# calculate how much no of notnull value present in overall dataset
print("total not null value in %  ",dataset.notnull().sum().sum()/(dataset.shape[0]*dataset.shape[1])*100)

print(dataset.shape)

print()

# show the graphical representation
import seaborn as sns
import matplotlib.pyplot as plt
# sns.heatmap(dataset.isnull())

plt.show()

