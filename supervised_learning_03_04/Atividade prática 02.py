import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def init_dataset():
    data = datasets.load_iris()
    return data


def training_data_separation(size=0.35, random=42):
    data_set = init_dataset()
    x_train, x_test, y_train, y_test = train_test_split(data_set.data, data_set.target, test_size=size,
                                                        random_state=random)
    return x_train, x_test, y_train, y_test


def compute(x_input, y_input, x_test, y_test, range_size=30, score=100):
    data_index = []
    data_accuracy = []
    for k in range(range_size):
        k = k + 1
        model_knn = KNeighborsClassifier(k)
        model_knn.fit(x_input, y_input)
        y_pred = model_knn.predict(x_test)
        data_index.append(k)
        acc = accuracy_score(y_test, y_pred) + score
        # print(k, ' ', acc)
        data_accuracy.append(acc)
    return data_index, data_accuracy


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = training_data_separation()
    index, accuracy = compute(x_train, y_train, x_test, y_test)

    plt.subplot(2, 1, 1)
    plt.plot(index, accuracy)
    plt.title('K x Accuracy')
    plt.xlabel('Value of K')
    plt.ylabel('Accuracy')
    plt.show()
