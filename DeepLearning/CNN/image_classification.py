import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets,layers,models

(X_train,Y_train),(X_test,Y_test)=datasets.cifar10.load_data()
print(X_train.shape)
print(X_test.shape)

# print(X_train[0])

# visual rep of image 0
plt.figure(figsize=(15,2))
plt.imshow(X_train[1])
plt.show()
