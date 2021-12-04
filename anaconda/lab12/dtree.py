import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


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


def featureGain(df, feature, HS, targetColumn):
    print("calculating gain for", feature)
    values, counts = np.unique(df[feature], return_counts=True)
    entropies = []
    total = df.shape[0]

    for i, value in enumerate(values):
        # print("{value} occurs {i} times in {feat}".format(value=value, i=counts[i], feat=feature))
        tempDf = df[df[feature] == value]
        _, ent = p(tempDf[targetColumn].values)
        ent = get_entropy(ent)
        entropies.append(ent)

    infoGain = 0
    for i, entropy in enumerate(entropies):
        infoGain += (counts[i] / total) * entropy

    return HS - infoGain


def bestFeature(df, targetColumn):
    # complete entropy of the dataset
    _, ps = p(df[targetColumn].values)
    HS = get_entropy(ps)
    print("HS (for whole dataset)", HS)
    print()

    features = list(df)[0:-1]  # list of feature names

    # gains for features
    choice = features[
        np.argmax([featureGain(df, feature, HS, targetColumn) for feature in features])
    ]
    print("best feature:", choice)

    return choice


def make_sub_tree(df, feature, classes, targetColumn):
    # get unique values for this feature
    values = df[feature].unique()

    # empty tree
    tree = {}

    value_counts = df[feature].value_counts(sort=False)
    for value, count in value_counts.iteritems():
        value_df = df[df[feature] == value]

        node = False

        for c in classes:
            class_count = value_df[value_df[targetColumn] == c].shape[0]

            if class_count == count:
                tree[value] = c
                newDf = df[df[feature] != value]
                node = True

        if not node:
            tree[value] = "?"

    return tree, newDf


def make_tree(df, root, prev, targetColumn):
    if df.shape[0] != 0:
        print()
        choiceFeature = bestFeature(df, targetColumn)

        classes = df[targetColumn].unique()

        # make subree
        tree, newdf = make_sub_tree(df, choiceFeature, classes, targetColumn)

        if prev:
            root[prev] = dict()
            root[prev][choiceFeature] = tree
            next_root = root[prev][choiceFeature]
        else:
            root[choiceFeature] = tree
            next_root = root[choiceFeature]

        for node, branch in list(next_root.items()):
            if branch == "?":
                n = newdf[newdf[choiceFeature] == node]
                print(next_root, tree)
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
df = pd.read_csv("Iris.csv")
df = df.iloc[:, 1:]
print(df)

trained_tree = id3(df)
print("tree: ", trained_tree)

# preds = predictSet(df.iloc[:,0:-1], trained_tree)
preds = predictSet(df, trained_tree)

score = 0
values = df.iloc[:, -1].values
for i, pred in enumerate(preds):
    if pred == values[i]:
        score += 1

print("acc:", score / len(values))

