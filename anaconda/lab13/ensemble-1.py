import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("Iris.csv")
array = df.values

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

seed = 10
num_trees = 15
print("Using Ada Boost Classifiers, with no. of trees = ", num_trees)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print("Accuracy = ", accuracy_score(y_pred, y_test))

y_true = y_test
print("\nConfusion Matrix: \n", confusion_matrix(y_true, y_pred))

matrix = classification_report(y_true, y_pred)
print("\nClassification report : \n", matrix)

