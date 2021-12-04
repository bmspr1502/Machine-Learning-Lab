import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics


def bagging(X, y):
    n_samples = X.shape[0]
    # doing random sampling with replacement
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X.iloc[indices], y.iloc[indices]


class BaggedClassifier:
    def __init__(self, n_estimators, n_neighbours=5):
        self.n_estimators = n_estimators
        self.n_neighbours = n_neighbours
        self.classifiers = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            clf = DecisionTreeClassifier(max_depth=4)

            # getting random sample for the given input
            X_sample, y_sample = bagging(X, y)

            # fitting the data on the given input
            clf.fit(X_sample, y_sample)

            self.classifiers.append(clf)

    def predict(self, X):
        preds = np.array([clf.predict(X) for clf in self.classifiers])
        preds = np.swapaxes(preds, 0, 1)

        # majority vote
        y_pred = [np.argmax(np.bincount(pred)) for pred in preds]
        return y_pred


dataset = load_iris()
df = pd.DataFrame(
    {
        "sepal length": dataset.data[:, 0],
        "sepal width": dataset.data[:, 1],
        "petal length": dataset.data[:, 2],
        "petal width": dataset.data[:, 3],
        "species": dataset.target,
    }
)
print("Dataset Example\n")
print(df.sample(5))

X = df.iloc[:, :4]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

print("\nBuilding random forest classifier")
clf = BaggedClassifier(n_estimators=50)
clf.fit(X_train, y_train)
print("number of classifiers:", clf.n_estimators)

y_pred = clf.predict(X_test)
print()
print("accuracy:", metrics.accuracy_score(y_test, y_pred))
print("confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))
print("Classification Report: \n", metrics.classification_report(y_test, y_pred))
