from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def init():
    iris = datasets.load_iris()
    print('Data: ', iris.data)
    print('Target: ', iris.target)
    print('Names: ', iris.target_names)
    print('Feature: ', iris.feature_names)

    return iris


def assembling_graphics(iris, x_index=0, y_index=1):
    plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
    plt.xlabel(iris.feature_names[x_index])
    plt.ylabel(iris.feature_names[y_index])

    plt.show()


def test_iris_data(iris):

    # Indices das features sendo plotadas
    assembling_graphics(iris)

    # Indices das features sendo plotadas
    assembling_graphics(iris, 2, 3)


def prediction_test(iris):

    model = GaussianNB()
    model.fit(iris.data, iris.target)

    GaussianNB(priors=None, var_smoothing=1e-09)

    data_test = [[5.2, 3.6, 1.4, 0.2],
                 [4.9, 3.0, 1.4, 0.2],
                 [4.7, 3.3, 1.3, 0.2],
                 [6.8, 2.9, 4.8, 1.4],
                 [6.7, 3.0, 3.5, 1.5],
                 [5.7, 3.6, 4.5, 1.0],
                 [5.5, 2.4, 1.0, 1.0]]

    pred = model.predict(data_test)
    print('Prediction: ', pred)


def calculate_accuracy(iris):

    model = GaussianNB()
    model.fit(iris.data, iris.target)

    model.fit(iris.data, iris.target)
    pred = model.predict(iris.data)
    accuracy = accuracy_score(iris.target, pred)
    print('Accuracy: ', accuracy)


if __name__ == "__main__":
    iris = init()
    test_iris_data(iris)
    prediction_test(iris)
    calculate_accuracy(iris)
