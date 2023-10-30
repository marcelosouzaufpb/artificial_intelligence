import matplotlib.pyplot as plt
import numpy as np

from supervised_learning_01_02.generating_fictitious_data import generate_data
from supervised_learning_01_02.iris_api import display_iris_data
from supervised_learning_01_02.iris_api import load_iris_data
from supervised_learning_01_02.iris_api import train_linear_regression


def plot_linear_regression(model, raw_x, raw_y, x_range):
    X_range = x_range[:, np.newaxis]
    predicted_y = model.predict(X_range)

    plt.scatter(raw_x, raw_y)  # Raw data
    plt.plot(x_range, predicted_y)  # Predicted values

    plt.show()


def main():
    x, y = generate_data()
    xfit = np.linspace(-1, 13)  # Generates well-distributed numbers between -1 and 13
    model = train_linear_regression(x, y)

    print('Coefficients: ', model.coef_)
    print('Intercept: ', model.intercept_)
    print('Predict: ', model.predict([[11.0]]))

    iris_data = load_iris_data()
    display_iris_data(iris_data)

    plot_linear_regression(model, x, y, xfit)


if __name__ == '__main__':
    main()
