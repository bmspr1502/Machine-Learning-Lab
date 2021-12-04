import numpy as np
import pandas as pd
from math import sqrt, exp, pi
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('./iris.csv')
print('training data set sample')
print(data.sample(5))

def labels_to_num(data):
    # convert strings to numbers
    data.replace(['Iris-versicolor','Iris-setosa','Iris-virginica'], [0,1,2], inplace=True)

print('\nreplaced labels with numbers')
labels_to_num(data)
print(data.head(5))

# split data by class
def split_by_class(data):
    # split data by class
    classes = dict()
    for row in data:
        if(row[-1] not in classes.keys()):
            classes[row[-1]] = list()
            
        classes[row[-1]].append(row)

    return classes

# get mean, std and size
def get_info(data):
    # get data
    info = [(np.mean(col), np.std(col), len(col)) for col in zip(*data)]
    del info[-1] # remove target coloumn
    return info

# get mean, std and size by class
def get_info_by_class(data):
    classData = split_by_class(data)
    info = dict()
    for classVal, rows in classData.items():
        info[classVal] = get_info(rows)
    return info

def calc_prob(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

def predict(info, dataset):
    outProbs = list()

    for data in dataset:
        total_rows = sum([info[label][0][2] for label in info])
        probs = dict()

        for classVal, class_info in info.items():
            probs[classVal] = info[classVal][0][2]/float(total_rows)
            for i in range(len(class_info)):
                mean, stdev, _ = class_info[i]
                probs[classVal] *= calc_prob(data[i], mean, stdev)

        outProbs.append(probs)

    preds = list()
    for prob in outProbs:
        preds.append(max(prob, key=prob.get))
    return preds

# split training and testing data
X_train, X_test = train_test_split(data,test_size=0.2)

info = get_info_by_class(X_train.values)
y_test = X_test.iloc[:,-1].values
X_test = X_test.iloc[:,0:-1]
preds = predict(info, X_test.values)
 
print('accuracy', metrics.accuracy_score(y_test,preds))
print('confusion matrix')
print(metrics.confusion_matrix(y_test, preds))
print('classification report')
print(metrics.classification_report(y_test, preds))
