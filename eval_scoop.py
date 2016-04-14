from sklearn.metrics import mean_squared_error
from timeit import default_timer as timer 
from scoop import futures

from rbf import *
from utils import class_accuracy
from data import preprocess_mnist, preprocess_digits 

def eval(p, X_train, Y_train, X_test, Y_test):
    model = RBFNet(1000, p=p)
    model.fit(X_train, Y_train)

    print("Predict train data")
    yy = model.predict(X_train)
    acc_train = class_accuracy(Y_train, yy)

    print("Predict test data")
    yy = model.predict(X_test)
    acc_test = class_accuracy(Y_test, yy)

    return model, acc_train, acc_test

if __name__ == "__main__":
    
    start = timer() 
    
    #(X_train, Y_train), (X_test, Y_test) = preprocess_digits() 
    (X_train, Y_train), (X_test, Y_test) = preprocess_mnist() 

    param_values = list( np.logspace(-1, -4, 4)  )
    ret_values = list(futures.map(lambda x: eval(x, X_train, Y_train, X_test, Y_test), param_values))
    
    for model, train_acc, test_acc in ret_values:
        print(model.get_params())
        print("Train_acc %s" % train_acc) 
        print("Test_acc %s" % test_acc)

    end = timer() 
    minutes = (end-start)/60 
    print("Time:", minutes ) 
