''' Neural Network From Scratch Exercise '''
import numpy as np
import pandas as pd
from matplotlib import pyplot
from typing import Tuple

m, n = None, None

def read_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''Reads mnist dataset and splits between training and test'''
    global m
    global n
    pd_data = pd.read_csv('./data/train.csv')
    data = np.array(pd_data)
    m, n = data.shape
    np.random.shuffle(data)
    holdout = data[0:1000].T
    y_holdout, x_holdout = holdout[0], holdout[1:n]

    train = data[1000:m].T
    y_train, x_train = train[0], train[1:n]
    x_holdout, x_train = x_holdout / 255., x_train / 255.
    print("Data successfully read")
    return y_train, x_train, y_holdout, x_holdout

def init_params() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5

    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return w1, b1, w2, b2


def softmax(z: np.ndarray) -> np.ndarray:
    with np.errstate(over='ignore'):
        return np.exp(z) / sum(np.exp(z))

def forward_prop(w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    relu = lambda z: np.maximum(0, z)

    z1 = w1.dot(x) + b1
    a1 = relu(z1)

    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)

    return z1, a1, z2, a2

def one_hot(y: np.ndarray) ->  np.ndarray:
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def back_prop(z1: np.ndarray, a1: np.ndarray, a2: np.ndarray, w2: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.int64, np.int64, np.int64, np.int64]:
    derivative_relu = lambda z: z > 0

    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2)
    # db2 = 1 / m * np.sum(dz2, axis=1).reshape(-1, 1)
    dz1 = w2.T.dot(dz2) * derivative_relu(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    # db1 = 1 / m * np.sum(dz1, axis=1).reshape(-1, 1)
    db1 = 1 / m * np.sum(dz1)

    return dw1, db1, dw2, db2
    

def update_params(w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, dw1: np.int64, db1: np.int64, dw2: np.int64, db2: np.int64, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def get_predictions(a2: np.ndarray) -> np.ndarray:
    return np.argmax(a2, 0)

def get_accuracy(predictions: np.ndarray, y: np.ndarray) -> np.float64:
    return np.sum(predictions == y) / y.size

def gradient_descent(x: np.ndarray, y: np.ndarray, iterations: int, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.float64]:
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = back_prop(z1, a1, a2, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if (i % 10) == 0:
            print(f"Iteration: {i}")
            print(f"Accuracy: {get_accuracy(get_predictions(a2), y)}")
    train_accuracy = get_accuracy(get_predictions(a2), y)
    return w1, b1, w2, b2, train_accuracy

def make_predictions(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    _, _, _, a2 = forward_prop(w1, b1, w2, b2, x)
    predictions = get_predictions(a2)
    return predictions

def test_prediction(index: int, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, x_train: np.ndarray, y_train: np.ndarray) -> None:
    current_image = x_train[:, index, None]
    prediction = make_predictions(x_train[:, index, None], w1, b1, w2, b2)
    label = y_train[index]
    print(f"Prediction: {prediction}\nLabel: {label}")

    current_image = current_image.reshape((28, 28)) * 255
    pyplot.gray()
    pyplot.imshow(current_image, interpolation='nearest')
    pyplot.show()

def main():
    y_train, x_train, y_test, x_test = read_data()
    w1, b1, w2, b2, train_accuracy = gradient_descent(x_train, y_train, 300, 0.05)
    print(f"Train accuracy: {train_accuracy}")

    test_prediction(0, w1, b1, w2, b2, x_train, y_train)
    test_prediction(1, w1, b1, w2, b2, x_train, y_train)
    test_prediction(2, w1, b1, w2, b2, x_train, y_train)

    predictions = make_predictions(x_test, w1, b1, w2, b2)
    print(f"Test accuracy: {get_accuracy(predictions, y_test)} ")

main()