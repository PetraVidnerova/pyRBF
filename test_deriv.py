import pandas as pd 
import numpy.random as rand 
from rbf import *
from sklearn.metrics import mean_squared_error
from data import preprocess_digits, preprocess_sin  
from utils import class_accuracy 
from timeit import default_timer as timer 
import sys

(X_train, Y_train), (X_test, Y_test) = preprocess_sin() 

start = timer() 
model = RBFNet(10, kernel="Gaussian", p=0.00001, set_centers="random", compute_widths='none')
model.fit(X_train, Y_train) 
yy = model.predict(X_train)
end = timer() 
print("Time: %ss" % (end-start))

start = timer() 
err = 0.5*100*mean_squared_error(Y_train, yy)
print("Err:", err)
acc = class_accuracy(Y_train, yy) 
print("Acc:", acc) 
end = timer() 
print("Time: %ss" % (end-start))

model._minimize(X_train, Y_train)
params = model._parameters()
params = np.random.random(len(params)) 
err = model._objective(params) 
print("Err: %s " % err)

deriv = model._derivative(params)

delta = 0.001
lenc = 250*X_train.shape[1] 
for i in range(len(params)):
    params2 = params.copy()
    params2[i] += delta 
    err2 = model._objective(params2)
    deriv_approx = (err2-err)/delta 
    print(deriv_approx, deriv[i]) 
    sys.stdout.flush() 

