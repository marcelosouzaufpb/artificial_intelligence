import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression


def load_iris_data():
    irisds = load_iris()
    return irisds


def display_iris_data(iris_data):
    print("Iris Data:", iris_data.data)
    print("Iris Feature Names:", iris_data.feature_names)
    print("Iris Target:", iris_data.target)
    print("Iris Target Names:", iris_data.target_names)


def train_linear_regression(x, y):
    model = LinearRegression(fit_intercept=True)
    X = x[:, np.newaxis]
    model.fit(X, y)
    return model


def predict_iris(model, sepal_length):
    return model.predict([[sepal_length]])
