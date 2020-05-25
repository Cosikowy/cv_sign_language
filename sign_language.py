# import keras
import pandas as pd
import numpy as np
import matplotlib as plt
# from keras.datasets import mnist

train = np.genfromtxt('./sign_language/sign_mnist_train.csv', skip_header=1, delimiter=',')

# print(train.shape)

# (x_train, y_train), (x_test, y_test) = mnist.load_data()


print(train.shape)