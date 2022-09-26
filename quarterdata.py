from tensorflow import keras
from keras import optimizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# MNISTデータを読込む
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=10000, random_state=2)

print(len(x_train))
print(len(x_valid))
print(len(x_test))
print(len(y_train))
print(len(y_valid))
print(len(y_test))

x_train, x_no, y_train, y_no = train_test_split(x_train, y_train, train_size=12500, random_state=2)

print(len(x_train))
print(len(x_valid))
print(len(x_test))
print(len(y_train))
print(len(y_valid))
print(len(y_test))

""" 
modelquarter = Sequential()

modelquarter.add(Conv2D(filters = 20, kernel_size = (5, 5), padding = 'Same', activation ='relu', input_shape = (28,28,1)))
modelquarter.add(MaxPooling2D(pool_size=(2, 2)))

modelquarter.add(Conv2D(filters = 50, kernel_size = (5, 5), padding = 'Same', activation ='relu'))
modelquarter.add(MaxPooling2D(pool_size=(2, 2)))

modelquarter.add(Conv2D(500,kernel_size =  (4, 4),padding = 'Same', activation ='relu'))
modelquarter.add(Conv2D(10,kernel_size =  (1, 1),padding = 'Same', activation ='relu'))
modelquarter.add(Flatten())
modelquarter.add(Dense(500, activation = "relu"))
modelquarter.add(Dense(10, activation = "softmax"))

import time

start = time.time()
opt = keras.optimizers.RMSprop(lr=1e-7, rho=0.9, epsilon=1e-08, decay=0.0)
modelquarter.compile(optimizer = opt, loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
history = modelquarter.fit(x_train_q, y_train_q, validation_data=(x_valid_q, y_valid_q), epochs=100, batch_size=8, verbose=1)
finish_time = time.time() - start

fig = plt.figure(figsize=(8, 8))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.text(0.38, 1.12, "Quarter data", fontsize="x-large")
plt.text(0.35, 1.08, "Learning time : " + f'{round(finish_time,5)}' + "sec", color="red")


#ヒストリーの可視化（正確）
fig.add_subplot(1, 2, 1) 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(acc, label="Training accuracy")
plt.plot(val_acc,label="Validation accuracy")
plt.title('modelquarter accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='best')

#ヒストリーの可視化（損失）
fig.add_subplot(1, 2, 2) 
loss = history.history["loss"]
val_loss = history.history["val_loss"]
plt.plot(loss, label="Training loss")
plt.plot(val_loss, label="validation loss")
plt.title('modelquarter loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='best')
plt.show()
 """