from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt


# Function to load the Iris dataset
def load_iris_dataset():
    iris = load_iris()
    x = iris.data
    y = iris.target
    return x, y


# Function to split the data into training and testing sets
def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)
    return x_train, x_test, y_train, y_test


# Function to train and evaluate a K-Nearest Neighbors classifier
def knn_classifier(x_train, y_train, x_test, y_test, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = knn.score(x_test, y_test)
    return accuracy


# Function to perform K-Fold cross-validation
def k_fold_cross_validation(x, y, n_splits, shuffle=False, random_state=None):
    kf = (n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return kf.split(x)


# Function to plot the cross-validation graph for different K values
def plot_cross_validation_accuracy(x, y, k_range):
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x, y, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())

    plt.plot(k_range, k_scores)
    plt.xlabel('K value for KNN')
    plt.ylabel('Cross Validation Accuracy')
    plt.show()


if __name__ == "__main__":
    x, y = load_iris_dataset()
    x_train, x_test, y_train, y_test = split_data(x, y)

    # Example of using the functions
    accuracy = knn_classifier(x_train, y_train, x_test, y_test, n_neighbors=6)
    print('Accuracy:', accuracy)

    k_range = range(1, 31)
    plot_cross_validation_accuracy(x, y, k_range)
