import pandas as pd
import numpy as np

dic={"A":np.random.randint(1,100,size=60),
     "B":np.random.randint(1,100,size=60),
     "C":np.random.randint(1,10,size=60),
     "D":np.random.randint(1,120,size=60),
     "E":np.random.randint(1,340,size=60),
     "F":np.random.randint(1,30,size=60),
     "G":np.random.randint(1,700,size=60),
     "H":np.random.randint(1,40,size=60),
     "I":np.random.randint(1,450,size=60),
     "J":np.random.randint(1,1340,size=60),
     "K":np.random.randint(1,450,size=60),
     "L":np.random.randint(1,350,size=60)


     }

df=pd.DataFrame(dic)
print(df)
df.to_csv("new_file1.csv")
