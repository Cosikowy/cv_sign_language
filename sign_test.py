# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import keras
import os
import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

cwd = os.getcwd()


# %%
path_train = cwd+'/sign_language/sign_mnist_train.csv'
path_test = cwd+'/sign_language/sign_mnist_test.csv'

y_train = np.genfromtxt(path_train, skip_header=1, delimiter=',', usecols=[0])
x_train = np.genfromtxt(path_train, skip_header=1, delimiter=',', usecols=[x for x in range(1,785)])
y_test = np.genfromtxt(path_test, skip_header=1, delimiter=',', usecols=[0])
x_test = np.genfromtxt(path_test, skip_header=1, delimiter=',', usecols=[x for x in range(1,785)])


print(y_train.shape)
print(x_train.shape)

print(y_test.shape)
print(x_test.shape)


# %%
x_train = x_train.reshape(27455, 28,28,1)
y_train = to_categorical(y_train)

x_test = x_test.reshape(7172, 28,28,1)
y_test = to_categorical(y_test)


# %%
x_test[0][0]


# %%

model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu',data_format='channels_last', input_shape=(28,28,1)))

model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(Flatten())

model.add(Dense(25, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)

