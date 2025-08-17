import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

df=pd.read_csv('insurance_data.csv')
# print(df.head(5))

inputs=df[['age','affordibility']]
output = df['bought_insurance']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(inputs,output,test_size=0.2,random_state=25)

# print(X_train)

# sclling
X_train_scale=X_train.copy()
X_train_scale['age']=X_train_scale['age']/100

X_test_scale=X_test.copy()
X_test_scale['age']=X_test_scale['age']/100

# print(X_test_scale)

# train nural nw
# kernel_initializer='ones' =w1 and w2
# bias_initializer='zeros'=bias
model=keras.Sequential([
    keras.layers.Dense(1,input_shape=(2,),activation='sigmoid',kernel_initializer='ones',bias_initializer='zeros')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_scale,Y_train,epochs=5)

# check the test accuracy
print(model.evaluate(X_test_scale,Y_test))


# print-weight and bias
coef,intercept=model.get_weights()
print(coef)
print(intercept)