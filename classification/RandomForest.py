import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits=load_digits()

print(dir(digits))


# --------------<>----------------
# plot the digit
# plt.gray()
# for i in range(4):
#     plt.matshow(digits.images[i])
#
# plt.show()


# --------------><-----------

# print the data in form of number array
# print(digits.data[:2])

print()
# independent variable
df=pd.DataFrame(digits.data)

# dependent variable add in df
df['target']=digits.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.drop(['target'],axis='columns'),digits.target,test_size=0.2)

print(len(X_train))

# use random forest classifier
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,y_train)


print("Accuracy of model :")
print(model.score(X_test,y_test))

print()
predicted_value=model.predict(X_test)

# plot confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predicted_value)
# print(cm)
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("truth")
plt.show()
