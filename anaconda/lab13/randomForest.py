import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics

def bagging(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True) # random sampling with replacement
    return X.iloc[indices], y.iloc[indices]

'''
    Random Forest Classifer
'''
class RandomForestClassifier:
    def __init__(self, n_trees=100):
        self.n_trees = n_trees
        self.num_features_for_split = None
        self.trees = []

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.num_features_for_split = int(math.sqrt(self.n_features))

        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(max_features=self.num_features_for_split) # max_features="sqrt" also works
            X_sample, y_sample = bagging(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0 , 1)

        # majority vote
        y_pred = [np.argmax(np.bincount(tree_pred)) for tree_pred in tree_preds]
        return y_pred

dataset = load_iris()
df = pd.DataFrame({
    'sepal length': dataset.data[:,0],
    'sepal width': dataset.data[:,1],
    'petal length': dataset.data[:,2],
    'petal width': dataset.data[:,3],
    'species': dataset.target
})

print('-----------DATASET-----------')
print(df.sample(5))

X = df.iloc[:,:4]
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print('\nBuilding random forest classifier')
clf = RandomForestClassifier(n_trees=100)
clf.fit(X_train, y_train)
print('number of trees:', clf.n_trees)
print('number of features:', clf.num_features_for_split)

y_pred = clf.predict(X_test)
print()
print('accuracy:', metrics.accuracy_score(y_test, y_pred))
print('confusion matrix:\n', metrics.confusion_matrix(y_test, y_pred))