import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("car_prices.csv")
print(df)
print()



plt.scatter(df["Mileage"],df["Sell Price"])
plt.show()

# independ
X=df[["Mileage","Age"]]
# dependent
Y=df["Sell Price"]


# model split for training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)

print(len(X_train))
print(len(X_test))

print("show train random 80% dataset ",X_train)


# train the model
from sklearn.linear_model import LinearRegression
clf=LinearRegression()

print()
clf.fit(X_train,Y_train)

# predict
print(clf.predict(X_test))

print()
print(Y_test)

# check accuracy
print(clf.score(X_test,Y_test))
