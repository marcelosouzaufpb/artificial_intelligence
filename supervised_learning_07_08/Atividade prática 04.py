import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier


# Load dataset
def load_data():
    datasets = load_breast_cancer()
    x = datasets.data
    y = datasets.target
    return x, y


# Question 1 - Plotting a Learning Curve
def plot_learning_curve(estimator, title, x, y, ylim=None, cv=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x, y, cv=cv, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.legend(loc="best")
    plt.show()


# Question 2 - Plotting a Validation Curve
def plot_validation_curve(estimator, param_name, param_range, x, y, title, alpha=0.1):
    train_scores, test_scores = validation_curve(estimator, x, y, param_name=param_name, param_range=param_range, cv=3)

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)

    plt.plot(param_range, np.mean(train_scores, axis=1), label="Training score", color="r")
    plt.fill_between(param_range, np.mean(train_scores, axis=1) - alpha, np.mean(train_scores, axis=1) + alpha,
                     alpha=0.2, color="r")

    plt.plot(param_range, np.mean(test_scores, axis=1), label="Cross-validation score", color="g")
    plt.fill_between(param_range, np.mean(test_scores, axis=1) - alpha, np.mean(test_scores, axis=1) + alpha, alpha=0.2,
                     color="g")

    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    x, y = load_data()

    est = DecisionTreeClassifier(max_depth=6)
    plot_learning_curve(est, "My Learning Curve", x, y, ylim=None, cv=6, train_sizes=np.linspace(.1, 1.0, 10))

    param_range = np.arange(1, 100)
    plot_validation_curve(est, param_name="max_depth", param_range=param_range, x=x, y=y, title="My Validation Curve",
                          alpha=0.1)
