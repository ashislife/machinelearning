import pandas as pd
import matplotlib.pyplot as plt

#  0-9 handwritten digits
from sklearn.datasets import load_digits

digits=load_digits()


# check what that digits contain attribute
print(dir(digits))

print()
# show 0-image in digit
print(digits.data[0])
print()

# --------------------->individual dependent and independent variable<----------------------------------
# #see the actual image of the numeric value of 0-5 in range
# plt.gray()
# for i in range(5):
#     plt.matshow(digits.images[i])
#
# plt.show()
#
# #target/dependent variable we want to predict
# print(digits.target[0:7])

#--------------------------<>---------------------------------------------------------------------------------

# train,test the model for whole digits available in library
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(digits.data,digits.target,test_size=0.2)

print(len(X_train))
print(len(X_test))


# train model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(X_train,Y_train)


# accuracy
print(model.score(X_test,Y_test))
print()
plt.matshow(digits.images[67])
plt.show()

# digits.target[67] → 67th image ka actual digit (0 to 9).
print(digits.target[67])


# if you want to know exactly its fail ,and get overall filling of model accuracy
# -------------confusion matrix-------------------------

y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_test,y_predicted)
# output shows in 2D array
print(cm)

# plot a graph of confusion matrix
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()
