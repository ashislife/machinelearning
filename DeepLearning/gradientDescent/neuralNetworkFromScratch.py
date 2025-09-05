
# ------------------->skippppppppppppppppppppppppppppppppppppppppppppppppppppppp<---------------------------------------------------

# import os
#
# from DeepLearning.lossOrCost_fun import y_predicted
#
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
# import matplotlib.pyplot as plt
#
# df=pd.read_csv('insurance_data.csv')
#
# inputs=df[['age','affordibility']]
# output = df['bought_insurance']
#
# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test=train_test_split(inputs,output,test_size=0.2,random_state=25)
#
# # print(X_train)
#
# # sclling
# X_train_scale=X_train.copy()
# X_train_scale['age']=X_train_scale['age']/100
#
# X_test_scale=X_test.copy()
# X_test_scale['age']=X_test_scale['age']/100
#
#
# def gradient_descent(age,affordability,y_true,epochs):
#     w1=w2=1
#     bias=0
#     rate=0.5
#     n=len(age)
#
#     for i in range(epochs):
#         weighted_sum=w1*age+w2*affordability+bias
#         y_predicted=sigmoid_numpy(weighted_sum)
#
#         loss=log_loss(y_true,y_predicted)
#         w1d=(1/n)*np.dot(np.transpose(age),(y_predicted-y_true))
#         w2d = (1 / n) * np.dot(np.transpose(affordability), (y_predicted-y_true))
#
#         bias_d=np.mean(y_predicted, y_true)
#
#         w1=w1-rate*w1d
#         w2=w2-rate*w2d
#         bias=bias-rate-bias_d
#
#         print(f'Epoch:{i}, w1:{w1},w2:{w2},bias:{bias},loss:{loss}')
#     return w1,w2,bias
#
# gradient_descent(X_train_scale['age'],X_test_scale['age'],y_train=1000)

