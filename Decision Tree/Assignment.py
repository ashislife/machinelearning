import pandas as pd

df=pd.read_csv("titanic.csv")
# print(df)

print()
# print(df.isnull().sum())
print()

finaldf=df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')
# print(finaldf)
print()
# print(finaldf.isnull().sum())

# fill the missing values
median_age=finaldf['Age'].median()
# print("Median of Age :",median_age)

finaldatset=finaldf.fillna(median_age)
print(finaldatset)
print(finaldatset.isnull().sum())

inputs=finaldatset.drop('Survived',axis='columns')
target=finaldatset['Survived']

# print(inputs)
# print(target)

# use onehotencoding (convert sex into numerical data)

from sklearn.preprocessing import LabelEncoder
le_Sex=LabelEncoder()

inputs['sex_n']=le_Sex.fit_transform(inputs['Sex'])
# print(inputs.head())

# new independent variable
new_input=inputs.drop('Sex',axis='columns')
print(new_input)


from sklearn import tree
model=tree.DecisionTreeClassifier()

model.fit(new_input,target)

print()