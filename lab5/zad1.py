import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")

# b
print(df)

# c
print(df.values)

# d 
# wszystkie wiersze, kolumna nr 0
print(df.values[:, 0])

# wiersze of 5 do 10, wszystkie kolumny
print(df.values[5:11, :])

# dane w komórce [1, 4]
print(df.values[1, 4])

# e
# podział na zbiór testowy (30%) i treningowy (70%), ziarno losowości = 13
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=13)

# f
print(test_set)
print(test_set.shape[0])

# g
train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]
