import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv("home_price.csv")
print("Home price : \n",df)

print()

# create a object for reg. and predict
model=linear_model.LinearRegression()

# train the model
model.fit(df[['area']],df.price)

predicted_price=model.predict([[5000]])
print("predicted price of 5000 sqft: ",predicted_price)


# save the trained model in your file
# it aloow to serialize python object into file

# --------------method1--------------------
import pickle
with open('model_pickle','wb')as f:
    # dump the model into the file
    pickle.dump(model,f)


# use the same model in read mode
with open('model_pickle','rb')as f:
    mp=pickle.load(f)

# use mp object to prediction
print("predicted price of 5000 sqft:",mp.predict([[5000]]))

# ----------------method2(used for large dataset/array)---------
import joblib

# direct save the model into file
joblib.dump(model,'model_joblib')

# use mj object to prediction
mj=joblib.load('model_joblib')
print("predicted price of 5000 sqft:",mj.predict([[5000]]))




