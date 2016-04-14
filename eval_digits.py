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

def eval_model(k, set_centers, compute_widths, p):
    global X_train, Y_train

    model = RBFNet(k, compute_widths=compute_widths, p=p, set_centers=set_centers, verbose=False)
    model.fit(X_train, Y_train)
    yy = model.predict(X_train)

    err = 100*mean_squared_error(Y_train,yy)
#    print("Err:", err)
    acc = class_accuracy(Y_train, yy)
#    print("Acc:", acc)
    
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


    K = [1000]
    P = [ 0.0001 ] 

    for p in P:
        err_list = []
        acc_list = [] 
        for err, acc  in list(futures.map(lambda x: eval_model(500, "random", "nearest_neighbor", p), range(16))):
            err_list.append(err)
            acc_list.append(acc)
        
        print("Nearest neighbor %s " % p)
        err_mean = sum(err_list)/len(err_list)
        acc_mean = sum(acc_list)/len(acc_list)
        print("Mean err: %s" % (err_mean))
        print("Mean acc: %s" % (acc_mean))  


        
    # plot errors 
#    plt.subplot("121")
#    random_plot, = plt.plot(K, random_err_means, "ro--", label="random") 
#    kmeans_plot, = plt.plot(K, kmeans_err_means, "go--", label="kmeans")
#    plt.legend(handles=[random_plot, kmeans_plot])
#    plt.xlabel("number of hidden units") 
#    plt.ylabel("mean error")
#    plt.title("Mean squarred error")
#
#    plt.subplot("122") 
#    random_plot, = plt.plot(K, random_acc_means, "ro--", label="random") 
#    kmeans_plot, = plt.plot(K, kmeans_acc_means, "go--", label="kmeans")
 #   plt.legend(handles=[random_plot, kmeans_plot])
 #   plt.xlabel("number of hidden units") 
 #   plt.ylabel("mean accuracy")
 #   plt.title("Classification accuracy")
#
 #   plt.savefig("kmeans_vs_random_fixed_widths.eps")


