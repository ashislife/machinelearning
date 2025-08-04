import pandas as pd

df=pd.read_csv("dummiesDataset.csv")
print(df)

print()
# convert into numerical from categorical
dummies=pd.get_dummies(df.Town,dtype=int)
print(dummies)

print()


# merge numerical data in original dataset
merge=pd.concat([df,dummies],axis="columns")
print(merge)


print()
# after converting ,drop the categorical data and any one dummy variable for the next process

final=merge.drop(['Town','west windsor'],axis="columns")
print(final)

print()

# import linear mode
from sklearn.linear_model import LinearRegression
model=LinearRegression()

# drop the price bc price->dependent variable
X=final.drop('Price',axis="columns")
print(X)

print()
# y-->dependent variable
Y=final.Price
print(Y)

# train the model
model.fit(X,Y)

# predict the price
print("predicted price :",model.predict([[2800,0,1]]))
print("predicted price :",model.predict([[3400,0,0]]))


# calculate the model prediction rate
print("Total Accuracy",model.score(X,Y))