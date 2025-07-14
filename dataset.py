import pandas as pd

dic={"A":[12,13,45,6,78],"b":[18,53,23,76,98],"c":[28,33,63,6,938]}

d=pd.DataFrame(dic)
print(d)
d.to_csv("new_file1.csv")
