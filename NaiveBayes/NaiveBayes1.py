import pandas as pd

df=pd.read_csv("titanic.csv")
# print(df.head(5))

final=df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')
# print(final.head(5))

target=final.Survived
input=final.drop('Survived',axis='columns')



# convert  sex into numerical
from sklearn.preprocessing import LabelEncoder
male=LabelEncoder()
female=LabelEncoder()
input['male_n']=male.fit_transform(input['Sex'])
input['female_n']=female.fit_transform(input['Sex'])

final_input=input.drop('Sex',axis='columns')

# print(final_input.head(5))

# check Nan value is present ?
print(final_input.columns[final_input.isna().any()])

# fill the NaN value
final_input.Age=final_input.Age.fillna(final_input.Age.mean())
print(final_input.head(10))

# tain test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(final_input,target,test_size=0.4)

# gaussian Naivebayes
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()

model.fit(X_train,Y_train)

print(model.score(X_test,Y_test))