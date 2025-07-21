import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv("sample_loan_data.csv")
print(dataset)

print()
print("missing values :",dataset.isnull().sum())

print()
print(dataset.info())
print()

print(dataset.select_dtypes(include="float64").columns)

print()


# esse v value fill ke sakte hai (mean value nikalkar NaN se replace ke dega)
from sklearn.impute import SimpleImputer
si=SimpleImputer(strategy="mean")
print(si.fit_transform(dataset[['ApplicantIncome']]))