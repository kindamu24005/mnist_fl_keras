from tensorflow import keras
from keras import optimizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten



def get_mnist_model():

    model = Sequential()

    model.add(Conv2D(filters = 20, kernel_size = (5, 5), padding = 'Same', activation ='relu', input_shape = (28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters = 50, kernel_size = (5, 5), padding = 'Same', activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(500,kernel_size =  (4, 4),padding = 'Same', activation ='relu'))
    model.add(Conv2D(10,kernel_size =  (1, 1),padding = 'Same', activation ='relu'))
    model.add(Flatten())
    model.add(Dense(500, activation = "relu"))
    model.add(Dense(10, activation = "softmax"))

    model.compile(
        optimizer=keras.optimizers.RMSprop(lr=1e-7, rho=0.9, epsilon=1e-08, decay=0.0),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    return model