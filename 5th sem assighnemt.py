import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# data = np.random.randn(5)
# plt.hist(data)
# plt.show()

# x=[1,2,3]
# y=[2,4,1]
# plt.plot(x,y)
# plt.scatter(x,y)
# plt.show()

# fig,axs=plt.subplots(2,2)
# axs[0,0].plot([1,2,3])
# axs[0,1].bar([1,2,3],[3,2,1])
# axs[1,0].bar([1,2],[2,1])
# axs[1,1].hist(np.random.randint(100))
# plt.tight_layout()
# plt.show()


df=pd.DataFrame(np.random.randn(10,3),columns=list('ABC'))
sns.heatmap(df.corr(),annot=True)
plt.show()