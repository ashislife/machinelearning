import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("insurance_data.csv")
print(df)

X=df[["age"]]
Y=df["bought_insurance"]

# plot a graph
plt.scatter(X,Y,color="r",marker="+")
plt.xlabel("Age")
plt.ylabel("bought_insurance")
plt.title("INSURANCE GRAPH ")
plt.show()

#test and split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1, random_state=10, stratify=Y)

print()
print("Train data is\n ",X_test)

print()

# train model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(X_train,Y_train)

print("Test dataset\n",model.predict(X_test))

# accuracy
print(model.score(X_test,Y_test))

# predict the probability
proba = model.predict_proba(X_test)
print(proba)