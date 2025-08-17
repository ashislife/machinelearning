
# sigmoid function
import math
def sigmoid(X):
    return 1/(1+math.exp(-X))
print(sigmoid(100))



# ---------------><----------------------

# tanh
import math

def tanh(X):
    return (math.exp(X) - math.exp(-X)) / (math.exp(X) + math.exp(-X))
print(tanh(100))



# ---------------><----------------------


# ReLU
def ReLU(X):
    return max(0,X)
print(ReLU(100))
print(ReLU(-5))


# ---------------><----------------------

# Leaky ReLU
def leakReLU(X):
    return max(0.1*X ,X)
print(leakReLU(-5))