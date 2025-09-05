import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# Ye system ke available devices (CPU/GPU) dikhayega
print(tf.config.experimental.list_physical_devices())

# check GPU available in our system or not
# print(tf.test.is_built_with_cuda())


# load the dataset
(X_train,Y_train),(X_test,Y_test)=tf.keras.datasets.cifar10.load_data()

print(X_train.shape)

# print(X_train[0].shape)




# plot the fig
# def plot_sample(index):
#     plt.figure(figsize=(10,3))
#     plt.imshow(X_train[index])
# plot_sample(0)
# plot_sample(1)
# plot_sample(3)
# plt.show()

# scaling the image
x_train_scale=X_train[0]/255
x_test_scale=X_test[0]/255

# scaling for hole data
X_train = X_train / 255.0
X_test  = X_test / 255.0


# show actual data(array form)
# print(Y_train[0:5])

# use one hot encoding(array form ko numerical dataset me convert karna )
Y_train_categorical=keras.utils.to_categorical(
    Y_train,num_classes=10
)
print(Y_train_categorical[0:5])


Y_test_categorical=keras.utils.to_categorical(
    Y_test,num_classes=10
)

# building model
model=keras.Sequential([
    # flatten(2d->1d)
    keras.layers.Flatten(input_shape=(32,32,3)),

    # hidden layers
    keras.layers.Dense(3000,activation='relu'),
    keras.layers.Dense(1000,activation='relu'),


    keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train,Y_train_categorical,epochs=50)

model.predict(X_test)