import pandas as pd

df=pd.read_csv("salaries.csv")
# print(df.head())

#independent and dependent variable
inputs=df.drop("salary_more_then_100k",axis="columns")
target=df['salary_more_then_100k']


# show the independent and target variable
print(inputs)
print()
print(target)

# use one hot encoding
from sklearn.preprocessing import LabelEncoder
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()

# convert categorical data into numerical (new extra columns)
inputs['company_n']=le_company.fit_transform(inputs['company'])
inputs['job_n']=le_company.fit_transform(inputs['job'])
inputs['degree_n']=le_company.fit_transform(inputs['degree'])

print(inputs.head())

print()
# drop the categogical data
input_n=inputs.drop(['company','job','degree'],axis='columns')
print(input_n)


# implement decision tree

from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(input_n,target)


print()
print("model accuracy")
print(model.score(input_n,target))

print()
print(model.predict([[2,0,1]]))