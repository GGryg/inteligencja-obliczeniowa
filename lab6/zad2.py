from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier
import pandas as pd

iris = pd.read_csv("iris.csv")

[train_set, test_set] = train_test_split(iris.values, train_size=0.7, random_state=278839)
# print(test_labels)

train_input = train_set[:, 0:4]
train_class = train_set[:, 4]
test_input = test_set[:, 0:4]
test_class = test_set[:, 4]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2, 3), random_state=278839)
clf.fit(train_input, train_class)
print(clf.predict(test_input))
print(clf.score(test_input, test_class))

# hidden_layer_size(2,)
# hidden_layer_size(3,)
# hidden_layer_size(2, 2, 2)
# hidden_layer_size(3, 3)