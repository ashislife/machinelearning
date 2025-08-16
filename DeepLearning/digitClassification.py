import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

# print(len(X_train))


# shape of 0
# print(X_train[0].shape)

# show the digit in array form
# print(X_train[0])

# plot the image
# plt.matshow(X_train[2])
# plt.show()

# print(Y_train[2])


# Scale values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Original shape
print(X_train.shape)  # (60000, 28, 28)

# Flatten
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)

# print(X_train_flattened.shape)  # (60000, 784)
# print(X_test_flattened.shape)   # (10000, 784)

# Model
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train_flattened, Y_train, epochs=5)

# accuracy
print(model.evaluate(X_test_flattened, Y_test))

print()

# predict
y_predicted=model.predict(X_test_flattened)
print(y_predicted[2])

# max val of prediction
np.argmax(y_predicted[2])

y_predicted_labels=[np.argmax(i) for i in y_predicted]

# confusion matrix
cm=tf.math.confusion_matrix(labels=Y_test,predictions=y_predicted_labels)
print(cm)

# plot graph
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('Truth')
plt.show()