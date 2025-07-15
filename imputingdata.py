import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv("loan_dataset.csv")
print(dataset)


print(dataset.dropna(inplace=True))
print(dataset.to_csv("check_output.csv", index=False))
print()
