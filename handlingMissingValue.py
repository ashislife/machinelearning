

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('loan_dataset.csv')
# print(dataset.head(4))
print("shape of dataset",dataset.shape)

print()
print(dataset.isnull().sum())
print()

print(sns.heatmap(dataset.isnull()))
plt.show()

print()

# +++++++++++++work on missing data -------------------------

# drop the particular row
print(dataset.drop(columns="Loan_Term_Months",inplace=True))

print()

print(dataset.isnull().sum())
print()

print("shape of dataset",dataset.shape)
print()
