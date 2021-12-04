import numpy as np
import pandas as pd
from sklearn import metrics

# classifier
from sklearn.tree import DecisionTreeClassifier

def sign(x):
    return abs(x)/x if x!=0 else 1

def I(flag):
    return 1 if flag else 0

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.clfs = [None] * n_estimators

    def fit(self, X, y):
        N = X.shape[0]
        X = np.float64(X)
        w = np.array([1/N] * N)

        for t in range(self.n_estimators):
            Gm = DecisionTreeClassifier(max_depth=1).fit(X,y,sample_weight=w).predict
                        
            errM = sum([w[i]*I(y[i]!=Gm(X[i].reshape(1,-1))) for i in range(N)])/sum(w)
            
            AlphaM = np.log((1-errM)/errM)
            
            w = [w[i]*np.exp(AlphaM*I(y[i]!=Gm(X[i].reshape(1,-1)))) for i in range(N)] 
            
            
            self.clfs[t] = (AlphaM,Gm)

    def predict(self, X):
        y = 0
        for m in range(self.n_estimators):
            AlphaM,Gm = self.clfs[m]
            y += AlphaM*Gm(X)
        signA = np.vectorize(sign)
        y = np.where(signA(y)==-1,-1,1)
        return y

from sklearn.datasets import make_classification
x, y = make_classification(n_samples=217)
y = np.where(y==0, -1,1)
X_train = X_test = x
y_train = y_test = y


print('\nBuilding ada boost classifier')
clf = AdaBoost(n_estimators=5)
clf.fit(X_train, y_train)
print('number of classifiers:', clf.n_estimators)

y_pred = clf.predict(X_test)
print()
print('accuracy:', metrics.accuracy_score(y_test, y_pred))
print('confusion matrix:\n', metrics.confusion_matrix(y_test, y_pred))