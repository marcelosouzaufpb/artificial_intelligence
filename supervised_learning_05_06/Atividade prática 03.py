from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# Load the breast cancer dataset
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target


# Question 01 - Analyze the effectiveness of the DecisionTreeClassifier with different values of the
# max_depth parameter (maximum tree depth).
def question_01():
    d_range = range(2, 10)
    mean_scores = []

    for d in d_range:
        model = DecisionTreeClassifier(max_depth=d)
        scores = cross_val_score(model, x, y, cv=10, scoring='accuracy')
        mean_scores.append(scores.mean())

    plt.plot(d_range, mean_scores)
    plt.xlabel('Value of max_depth for Tree')
    plt.ylabel('Cross-Validation Accuracy')
    plt.show()


# Question 02 - Repeat the experiment from the previous question, but instead of specifying an integer value for cv,
# specify a custom splitter.
def question_02():
    d_range = range(2, 10)
    mean_scores = []
    custom_cv = StratifiedKFold(n_splits=10, shuffle=False, random_state=None)

    for d in d_range:
        model = DecisionTreeClassifier(max_depth=d, splitter='random', random_state=42)
        scores = cross_val_score(model, x, y, cv=custom_cv, scoring='accuracy')
        mean_scores.append(scores.mean())

    plt.plot(d_range, mean_scores)
    plt.xlabel('Value of max_depth for Tree')
    plt.ylabel('Cross-Validation Accuracy')
    plt.show()


if __name__ == "__main__":
    # Run either question_01() or question_02() based on your preference
    question_02()
