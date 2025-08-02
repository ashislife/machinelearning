import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv("home_price.csv")
print("Home price : \n",df)

print()
plt.xlabel("area")
plt.ylabel("price")
plt.scatter(df.area,df.price,color="red",marker="+")
plt.show()

# create a object for reg. and predict
reg=linear_model.LinearRegression()

# train the model
reg.fit(df[['area']],df.price)

predicted_price=reg.predict([[3300]])
print("predicted price of 3300 sqft: ",predicted_price)



