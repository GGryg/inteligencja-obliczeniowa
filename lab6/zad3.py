from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd

diabetes = pd.read_csv("diabetes.csv")

[train_set, test_set] = train_test_split(diabetes.values, train_size=0.7, random_state=278839)
# print(test_labels)

train_input = train_set[:, 0:8]
train_class = train_set[:, 8]
test_input = test_set[:, 0:8]
test_class = test_set[:, 8]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(6, 3), random_state=278839, activation="relu", max_iter=500)
clf.fit(train_input, train_class)
print(clf.score(test_input, test_class))

y_true = test_class
y_pred = clf.predict(test_input)

print(confusion_matrix(y_true, y_pred))

# hidden_layer_size(2,)
# hidden_layer_size(3,)
# hidden_layer_size(2, 2, 2)
# hidden_layer_size(3, 3)