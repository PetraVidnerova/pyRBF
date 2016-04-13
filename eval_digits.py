from rbf import *
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_digits
from keras.utils import np_utils
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt 

from utils import class_accuracy
from data import preprocess_digits 

from scoop import futures 

# For digits test and train are same datasets.  
(X_train, Y_train), (X_test, Y_test) = preprocess_digits() 

def eval_model(k,set_centers):
    global X_train, Y_train

    model = RBFNet(k, compute_widths=True, p=0.0001, set_centers=set_centers)
    model.fit(X_train, Y_train)
    yy = model.predict(X_train)

    err = 100*mean_squared_error(Y_train,yy)
    print("Err:", err)
    acc = class_accuracy(Y_train, yy)
    print("Acc:", acc)
    
    return err, acc

# scoop fails when called from function 
#def eval_mean(set_centers):
#    err_list = []
#    acc_list = [] 
#    for err, acc  in list(futures.map(lambda x: eval_model(set_centers), range(100))):
#        err_list.append(err)
#        acc_list.append(acc)
#        
#    print(set_centers)
#    print("Mean err: %s" % sum(err_list)/len(err_list))
#    print("Mean acc: %s" % sum(acc_list)/len(acc_list))  
        


if __name__ == "__main__":

    random_err_means = [] 
    random_acc_means = []
    kmeans_err_means = [] 
    kmeans_acc_means = []
    K = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500]

    for k in K:
        err_list = []
        acc_list = [] 
        for err, acc  in list(futures.map(lambda x: eval_model(k,"random"), range(10))):
            err_list.append(err)
            acc_list.append(acc)
        
        print("Random")
        err_mean = sum(err_list)/len(err_list)
        acc_mean = sum(acc_list)/len(acc_list)
        print("Mean err: %s" % (err_mean))
        print("Mean acc: %s" % (acc_mean))  
        random_err_means.append(err_mean) 
        random_acc_means.append(acc_mean) 

        err_list = []
        acc_list = [] 
        for err, acc  in list(futures.map(lambda x: eval_model(k,"kmeans"), range(10))):
            err_list.append(err)
            acc_list.append(acc)
        
        print("Kmeans")
        err_mean = sum(err_list)/len(err_list)
        acc_mean = sum(acc_list)/len(acc_list)
        print("Mean err: %s" % (err_mean))
        print("Mean acc: %s" % (acc_mean))  
        kmeans_err_means.append(err_mean) 
        kmeans_acc_means.append(acc_mean) 
        
    # plot errors 
    random_plot, = plt.plot(K, random_err_means, "ro--", label="random") 
    kmeans_plot, = plt.plot(K, kmeans_err_means, "go--", label="kmeans")
    plt.legend(handles=[random_plot, kmeans_plot])
    plt.xlabel("number of hidden units") 
    plt.ylabel("mean error")
    plt.savefig("kmeans_vs_random.eps")


