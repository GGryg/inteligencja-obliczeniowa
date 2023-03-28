from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import graphviz

df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=278839)

# print(test_set)

test_class = test_set[:, 4]
test_input = test_set[:, 0:4]
train_class = train_set[:, 4]
train_input = train_set[:, 0:4]

clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=2, random_state=278839)
clf = clf.fit(train_input, train_class)
print(clf.score(test_input, test_class))

tree.plot_tree(clf)

good_predictions = 0
leng = test_set.shape[0]

for i in range(leng):
    if clf.predict([test_input[i]]) == test_class[i]:
        good_predictions += 1

print(good_predictions)
print(good_predictions/leng*100, "%")

y_true = test_class
y_pred = clf.predict(test_input)

print(confusion_matrix(y_true, y_pred))