
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits



# preform logistic reg(after reexecute accuracy will change)
digits=load_digits()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(digits.data,digits.target,test_size=0.3)
model=LogisticRegression()
model.fit(X_train,Y_train)
print("Logistic Regression Accuracy")
print(model.score(X_test,Y_test))
print()


# preform SVM(after reexecute accuracy will change)
svm=SVC()
svm.fit(X_train,Y_train)
print("SVM Accuracy")
print(svm.score(X_test,Y_test))


# use random forest(after reexecute accuracy will change)
rf=RandomForestClassifier()
rf.fit(X_train,Y_train)
print("Random forest Accuracy")
print(rf.score(X_test,Y_test))



# ---------------------<>----------------------------------------------
# use kFold
from sklearn.model_selection import KFold
kf=KFold(n_splits=3)
print(kf)

# show how data train using split in kfold
for train_index,test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index,test_index)


# ---------------------<>----------------------------------------------

#By default, it uses 5-fold cross-validation (cv=5).
# (after reexecute accuracy will same)
from sklearn.model_selection import cross_val_score
model1=cross_val_score(LogisticRegression(),digits.data,digits.target)
print(model1)

from sklearn.model_selection import cross_val_score
svc1=cross_val_score(SVC(),digits.data,digits.target)
print(svc1)