import pandas as pd
from sklearn.datasets import load_iris

iris=load_iris()
# directary of iris
print(dir(iris))

print()
# feature of flower
print(iris.feature_names)



print()

df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head(5))

print()

df['target']=iris.target
print(df.head(5))

print()

import matplotlib.pyplot as plt
df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]

print(df2.head(5))


# plot graph for sepal
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='g',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='b',marker='+')

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()


plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.show()

# plot graph for petal
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='g',marker='+')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='b',marker='+')


X=df.drop('target',axis='columns')
Y=df['target']
# train model
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4,random_state=10)
# print(len(Y_test))


# use SVM and import support vector classifier
from sklearn.svm import SVC
model=SVC(C=10)
model.fit(X_train,Y_train)


print()
print("Accuracy of model")
print(model.score(X_test,Y_test))
