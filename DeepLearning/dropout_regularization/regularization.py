import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv("sonar_dataset.csv",header=None)
# print(df.sample(5))

print("Shape of dataset ",df.shape)

# col containing null or not
# print(df.isna().sum())


# define dependent and independent variable
X=df.drop(60,axis='columns')
Y=df[60]

# convert the y-categorical into numerical(r-->1,m-->0)
y=pd.get_dummies(Y,drop_first=True).astype(int)
# print(y.sample(5))


print()
# count value 0/1
# print(y.value_counts())

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.25,random_state=1)

# find the shape
print(X_train.shape)
print(X_test.shape)


import tensorflow as tf
from tensorflow import keras
model=keras.Sequential([
    keras.layers.Dense(60,input_dim=60,activation='relu'),
    keras.layers.Dense(30,activation='relu'),
    keras.layers.Dense(15,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid'),

])
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.fit(X_train,Y_train,epochs=100,batch_size=8)

# check the actual accuracy
print(model.evaluate(X_test,Y_test))

# overfitting problrm are occours

# predicted value is float formate
y_pred=model.predict(X_test).reshape(-1)
print(y_pred[:10])

# convert predicted value float into integer(0/1) bc your test data in integer
y_pred=np.round(y_pred)
print(y_pred[:10])


# print precision , recall ,f1-score and accuracy
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(Y_test,y_pred))

# use droput to drop neuron (accuracy is low but test accuracy is good)

model=keras.Sequential([
    keras.layers.Dense(60,input_dim=60,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1,activation='sigmoid'),

])
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.fit(X_train,Y_train,epochs=100,batch_size=8)
# check the actual accuracy
print(model.evaluate(X_test,Y_test))