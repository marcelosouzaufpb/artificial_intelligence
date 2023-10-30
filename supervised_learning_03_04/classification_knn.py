from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def train_knn_model(data, labels, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data, labels)
    return knn


def evaluate_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy


if __name__ == "__main__":
    iris = datasets.load_iris()

    limit = 140
    x_train_part = iris.data[:limit]
    y_train_part = iris.target[:limit]

    x_test_part = iris.data[limit:]
    y_test_part = iris.target[limit:]

    k = 8
    trained_model_part = train_knn_model(x_train_part, y_train_part, k)
    accuracy_part = evaluate_model(trained_model_part, x_test_part, y_test_part)

    print('Accuracy for the partial dataset: ', accuracy_part)


if __name__ == "__main__":
    iris = datasets.load_iris()

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.35, random_state=42)

    k = 8
    trained_model_full = train_knn_model(x_train, y_train, k)
    accuracy_full = evaluate_model(trained_model_full, x_test, y_test)

    print('Accuracy for the full dataset: ', accuracy_full)
