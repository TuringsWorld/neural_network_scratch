''' Neural Network From Scratch Exercise '''
import numpy as np
import pandas as pd
from matplotlib import pyplot

m, n = None, None

def read_data():
    global m
    global n
    pd_data = pd.read_csv('./data/train.csv')
    data = np.array(pd_data)
    m, n = data.shape
    np.random.shuffle(data)
    test = data[0:1000].T
    y_test, x_test = test[0], test[1:n]

    data_train = data[1000:m].T
    y_train, x_train = data_train[0], data_train[1:n]
    return y_train, x_train, y_test, x_test
    print("Data successfully read")

def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5

    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return w1, b1, w2, b2


def softmax(z):
    with np.errstate(over='ignore'):
        return np.exp(z) / sum(np.exp(z))

def forward_prop(w1, b1, w2, b2, x):
    relu = lambda z: np.maximum(0, z)

    z1 = w1.dot(x) + b1
    a1 = relu(z1)

    z2 = w2.dot(a1) + b2
    print("got z2")
    print(z2)
    print(z2.shape)
    a2 = softmax(z2)

    return z1, a1, z2, a2

def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def back_prop(z1, a1, z2, a2, w2, x, y):
    derivative_relu = lambda z: z > 0

    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2, axis=1).reshape(-1, 1)
    dz1 = w2.T.dot(dz2) * derivative_relu(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1, axis=1).reshape(-1, 1)

    return dw1, db1, dw2, db2
    

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def get_predictions(a2):
    return np.argmax(a2, 0)

def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size

def gradient_descent(x, y, iterations, alpha):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        print(i)
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if (i % 10) == 0:
            print(f"Iteration: {i}")
            print(f"Accuracy: {get_accuracy(get_predictions(a2), y)}")
        return w1, b1, w2, b2
    
def main():
    y_train, x_train, y_test, x_test = read_data()
    w1, b1, w2, b2 = gradient_descent(x_train, y_train, 200, 0.1)

main()