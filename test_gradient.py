import pandas as pd 
import numpy.random as rand 
from rbf import *
from sklearn.metrics import mean_squared_error
from data import preprocess_digits, preprocess_sin  
from utils import class_accuracy 
from timeit import default_timer as timer 
import sys
from scoop import futures 



def evaluate():
    """ Run learning and returns error, classification accuracy, and time. """
    (X_train, Y_train), (X_test, Y_test) = preprocess_digits() 

    start = timer() 
    model = RBFNet(500, kernel="Gaussian", p=0.01, set_centers="random", compute_widths='none',
                   compute_weights = True, gradient = True)
    model.fit(X_train, Y_train) 
    yy = model.predict(X_train)
    end = timer() 

    err = 0.5*100*mean_squared_error(Y_train, yy)
    acc = class_accuracy(Y_train, yy) 


    return err, acc, (end-start) 


def main(): 

    ret_values = list(futures.map(lambda x: evaluate(), list(range(10))))
    
    errors = [] 
    accs = [] 
    for err, acc, time in ret_values:
        print("Err: %s Acc: %s, Time: %s" % (err, acc, time))
        errors.append(err)
        accs.append(acc) 

    print("Mean err: %s Mean acc: %s" % (np.mean(errors), np.mean(accs)))
        

if __name__=="__main__":
    main()
