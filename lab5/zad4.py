from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import pandas as pd

df = pd.read_csv("iris.csv")

[train_set, test_set] = train_test_split(df.values, train_size=0.7, random_state=278839)

test_class = test_set[:, 4]
test_input = test_set[:, 0:4]
train_class = train_set[:, 4]
train_input = train_set[:, 0:4]

clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(train_input, train_class)

print(clf.score(test_input, test_class)*100)

y_true = test_class
y_pred = clf.predict(test_input)

print(confusion_matrix(y_true, y_pred))

# 3, 5 i 11 nie różnią się

gnb = GaussianNB()

y_pred = gnb.fit(train_input, train_class).predict(test_input)

print("Number of mislabeled points out of a total %d points : %d" % (test_set.shape[0], (test_class != y_pred).sum()))
print(42/45*100)
print(confusion_matrix(y_true, y_pred))

# Najlepiej wypadają k-najbliższych sąsiadów