import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('diabetes.csv')
target_column = ['class'] 
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()

X = df[predictors].values
y = df[target_column].values

y = pd.get_dummies(y.flatten()).to_numpy()

model = Sequential()
model.add(Dense(6, activation='relu', input_dim=8))
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(X, y, validation_split = 0.3, epochs=500, batch_size=32)

pred_test= model.predict(X)
scores2 = model.evaluate(X, y, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))    
confusion_matrix(y.argmax(axis=1), pred_test.argmax(axis=1))

history.history.keys()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()