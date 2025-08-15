import pandas as pd

df=pd.read_csv("spam.csv")
# print(df.head(5))

print()

# check no. of scam or ham
print(df.groupby('Category').describe())

print()
# convert category into number
df['spam']=df['Category'].apply(lambda x:1 if x=='spam' else 0)
print(df.head(5))

# train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(df.Message,df.spam,test_size=0.3)







# ----------------------------------------<>----------------------------------------------------------
# convert message into count(digits)
from sklearn.feature_extraction.text import CountVectorizer
v=CountVectorizer()
X_train_count=v.fit_transform(X_train.values)

# use naive bayes
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train_count,Y_train)

emails=[
    'Ffffffffff. Alright no way I can meet up with you sooner?',
    'For fear of fainting with the of all that housework you just did? Quick have a cuppa',
    'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C'
]
emails_count=v.transform(emails)
print(model.predict(emails_count))
# ------------------------------><---------------------------


# use pipeline (pipeline simple kr dena hai work ko) digits me conver karne ke liye jyaada code nhi likhna rahta
from sklearn.pipeline import Pipeline
clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])
clf.fit(X_train,Y_train)
print(clf.score(X_test,Y_test))

print(clf.predict(emails))