import numpy as np 
from sklearn.datasets import load_digits
from keras.datasets import mnist 
from keras.utils import np_utils
import pandas as pd 

nb_classes = 10 

def preprocess_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return (X_train, Y_train), (X_test, Y_test) 


def preprocess_digits():
    digits = load_digits()

    X_train = digits.data 
    y_train = digits.target

    print(X_train.shape)
    print(y_train.shape)

    Y_train = np_utils.to_categorical(y_train, nb_classes)


    return (X_train, Y_train), (X_train, Y_train)


def preprocess_sin():
    data = pd.read_csv("sin.csv",";") 
    X = data[data.columns[:-1]] 
    y = data[data.columns[-1]]


    X = X.as_matrix()
    y = y.as_matrix()
    y = y.reshape((len(y),1))
    return (X, y), (X, y) 
