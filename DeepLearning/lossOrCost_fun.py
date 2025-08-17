import numpy as np
y_predicted=np.array([1,1,0,0,1])
y_true=np.array([0.30,0.7,1,0,0.5])

def mae(y_true,y_predicted):
    total_error=0
    for yt,yp in zip(y_true,y_predicted):
        total_error+=abs(yt-yp)
    print("Total Error",total_error)
    mae=total_error/len(y_true)
    return mae
print(mae(y_true,y_predicted))


print()

# same code using numpy
MAE=np.mean(np.abs(y_predicted-y_true))
print("mean square error",MAE)

total_error=np.sum(np.abs(y_predicted-y_true))
print("Total Sum",total_error)