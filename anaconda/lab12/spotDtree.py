import pandas as pd
import numpy as np

class Node:
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value
        self.next = None
        self.children = None

def p(data):
    unique, count = np.unique(data, return_counts=True)
    return unique, count / len(data)

def get_entropy(ps):
    return sum([(-p * np.log2(p)) for p in ps])


def getGini(ps):
    return sum([(p*p) for p in ps])


def giniIndex(df, feature, targetColumn):
    values, counts = np.unique(df[feature], return_counts=True)

    total = df.shape[0]
    ginis = []
    for i, value in enumerate(values):
        tempDf = df[df[feature] == value]
        _, ps = p(tempDf[targetColumn].values)
        gini = getGini(ps)
        ginis.append(1 - gini)

    return sum(ginis)

def bestFeature(df, targetColumn):
    features = list(df)[0:-1] # list of feature names

    # gains for features
    choice = features[np.argmin([giniIndex(df, feature, targetColumn) for feature in features])]
    print('best feature:', choice)

    return choice

def make_sub_tree(df, feature, classes, targetColumn):
    # empty tree
    tree = {}

    value_counts = df[feature].value_counts(sort=False)
    for value, count in value_counts.iteritems():
        value_df = df[df[feature] == value]

        node = False

        for c in classes:
            class_count = value_df[value_df[targetColumn] == c].shape[0]
            
            if(class_count == count):
                tree[value] = c
                newDf = df[df[feature] != value]
                node = True

        if not node:
            tree[value] = '?'

    return tree, newDf

def make_tree(df, root, prev, targetColumn):
    if(df.shape[0] != 0):
        print()
        choiceFeature = bestFeature(df, targetColumn)

        classes = df[targetColumn].unique()

        # make subree
        tree, newdf = make_sub_tree(df, choiceFeature, classes, targetColumn)

        if(prev):
            root[prev] = dict()
            root[prev][choiceFeature] = tree
            next_root = root[prev][choiceFeature]
        else:
            root[choiceFeature] = tree
            next_root = root[choiceFeature]


        for node, branch in list(next_root.items()):
            if(branch == '?'):
                n = newdf[newdf[choiceFeature] == node]
                make_tree(n, next_root, node, targetColumn)

def id3(df):
    tree = {}
    targetColumn = list(df)[-1]
    make_tree(df, tree, None, targetColumn)
    print()

    return tree

def predict(instance, tree):
    if not isinstance(tree, dict):
        return tree

    root_node = next(iter(tree))
    value = instance[root_node]

    if value in tree[root_node]:
        return predict(instance, tree[root_node][value])
    else:
        return None

def predictSet(df, tree):
    preds = []
    for i in range(0, df.shape[0]):
        # print(df.iloc[i, 0:-1])
        pred = predict(df.iloc[i, 0:-1], tree)
        preds.append(pred)
    return preds

# load dataset
df = pd.read_csv('./data/student.csv')
# df = df.iloc[:, 1:]
print(df)

trained_tree = id3(df)
print('tree: ', trained_tree)

preds = predictSet(df, trained_tree)

score = 0
values = df.iloc[:,-1].values
for i, pred in enumerate(preds):
    if(pred == values[i]):
        score += 1

print('acc:', score/len(values))