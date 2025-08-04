import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def gradient_descent(x,y):
    m=b=0
    learning_rate=0.0001
    n=len(x)
    iteration=1000

    for i in range(iteration):
        y_predicted = m * x + b
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])

        # deviation
        md = -(2 / n) * sum(x * (y - y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)

        # learning rate
        m = m - learning_rate * md
        b = b - learning_rate * bd

        print("m {} ,b {},cost {},iteration {}".format(m, b, cost, i))
    plt.scatter(x, y, color='r',label='actual data ')
    plt.plot(x, m * x + b, color='blue',label='regression line')
    plt.xlabel("math")
    plt.ylabel("cs")
    plt.title("Regression graph")
    plt.legend()
    plt.show()


df = pd.read_csv("marks.csv")
x = np.array(df['math'])
y = np.array(df['cs'])
gradient_descent(x,y)
