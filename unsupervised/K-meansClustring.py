import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv('income.csv')
# print(df.head(5))
plt.scatter(df['Age'],df['Income($)'])
# plt.show()

# use k means cluster

from sklearn.cluster import KMeans


km=KMeans(n_clusters=3)

y_predicted=km.fit_predict(df[['Age','Income($)']])
# cluster divided into (0,1,2)
print(y_predicted)


# show the cluster in new column
df['cluster']=y_predicted
print(df.head(4))
df0=df[df.cluster==0]
df1=df[df.cluster==1]
df2=df[df.cluster==2]

plt.scatter(df0.Age,df0['Income($)'],color='g')
plt.scatter(df1.Age,df1['Income($)'],color='r')
plt.scatter(df2.Age,df2['Income($)'],color='b')

plt.xlabel("Age")
plt.ylabel("Income")
plt.show()













# data are not grouping so we use(MinMaxScaler)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

scaler.fit(df[['Income($)']])
df['Income($)']=scaler.transform(df[['Income($)']])

scaler.fit(df[['Age']])
df['Age']=scaler.transform(df[['Age']])

km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['Age','Income($)']])
# cluster divided into (0,1,2)
print(y_predicted)
df['cluster']=y_predicted
# df.drop('cluster',axis='columns',inplace=True)





# plot a graph
df0=df[df.cluster==0]
df1=df[df.cluster==1]
df2=df[df.cluster==2]

plt.scatter(df0.Age,df0['Income($)'],color='g')
plt.scatter(df1.Age,df1['Income($)'],color='r')
plt.scatter(df2.Age,df2['Income($)'],color='b')

plt.xlabel("Age")
plt.ylabel("Income")
plt.show()