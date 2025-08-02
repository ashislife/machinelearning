import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math

df=pd.read_csv("homeprice.csv")
print(df)
print()

# calculate median for missing value
median_bedroom=math.floor(df.bedroom.median())

# fill missing value
df.bedroom=df.bedroom.fillna(median_bedroom)

# after fill dataset
print(df)

print()

reg=linear_model.LinearRegression()
reg.fit(df[['area','bedroom','age']],df.price)

# print gradient and intercept
print(reg.coef_)
print(reg.intercept_)

print()

print("predicted price ",reg.predict([[3000,3,40]]))





