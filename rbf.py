from sklearn.base import BaseEstimator, RegressorMixin
import scipy.linalg as linalg 
import numpy as np 
from hidden_layer import *

class RBFNet(BaseEstimator,RegressorMixin): 
 
    def __init__(self, k=10, p=1.0, compute_widths=False):
        self.k = k
        self.p = p
        self.compute_widths = compute_widths


    def fit(self,X,y):
        self.hl = HiddenLayer(self.k, p=self.p, compute_widths=self.compute_widths) 
        # Computes hidden layer actiovations.
        self.hidden_ = self.hl.fit_transform(X)
        # Computes output layer weights. 
        print("Solving ouput weights.")
        self.w_ = np.dot(linalg.pinv2(self.hidden_),y) 
        return self 

        
    def predict(self,X):
        self.hidden_ = self.hl.transform(X) 
        return np.dot(self.hidden_, self.w_) 
