import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv("india_budget.csv")
print(df)
print()


reg=linear_model.LinearRegression()

reg.fit(df[['year']],df.per_capita_income)


predicted_output=reg.predict([[2025]])
print("per capita income in 2025 in India :",predicted_output)

plt.xlabel("year")
plt.ylabel("Per_capita_income")
plt.scatter(df.year,df.per_capita_income,color='red',marker="+")

# draw best fit line
plt.plot(df.year,reg.predict(df[['year']]),color='blue')
plt.show()

