""" from tensorflow import keras
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
        metrics=["accuracy"],
    )

    return model

print(get_mnist_model().summary()) """

from copy import deepcopy
import tensorflow as tf
from tensorflow import keras

from stadle import AdminAgent
from stadle import BaseModelConvFormat
from stadle.lib.entity.model import BaseModel


def get_mini_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(5, activation='relu', input_shape=(3,)),
        keras.layers.Dense(4)
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    return model


model = get_mini_model()

# print(model)

# copy処理をする場合
#import copy
#model = copy.deepcopy(model)

model1 = model
print(id(model))
print(id(model1))
print(id(model1)==id(model))

model2= keras.models.clone_model(model)

print(id(model))
print(id(model2))
print(id(model2)==id(model))

model_weight = model.get_weights()

# model_weight = model.set_weights(model_weight)

# print(model_weight)

d = dict()

for i, w in enumerate(model_weight):
    d[f'layer_{i}'] = w

print(d)