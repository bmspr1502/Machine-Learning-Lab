import steph

tree = steph.dtree()
party, classes, features = tree.read_data("student.csv")
t = tree.make_tree(party, classes, features)
tree.printTree(t, " ")

print(tree.classifyAll(t, party))

for i in range(len(party)):
    tree.classify(t, party[i])


print("True Classes")
print(classes)
