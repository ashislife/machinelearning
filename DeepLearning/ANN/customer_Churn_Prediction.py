# skip (for some times )
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd

df=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.sample(5))

df.drop('customerID',axis='columns',inplace=True)
print(df.dtypes)

# check the value datatype
print(df.TotalCharges.values)

# convert string value into integer
print(pd.to_numeric(df.TotalCharges))