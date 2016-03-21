from sklearn.base import BaseEstimator, RegressorMixin
import scipy.linalg 
import numpy as np 
from hidden_layer import *

class RBFNet(BaseEstimator,RegressorMixin): 
 
    def __init__(self,k,p):
        self.hl = HiddenLayer(k,p) 

    def fit(self,X,y):
        # Computes hidden layer actiovations.
        self.hidden_ = self.hl.fit_transform(X)
        # Computes output layer weights. 
        self.w_ = np.dot(linalg.pinv2(self.hidden),y) 
        return self 

        
    def predict(self,X):
        self.hidden_ = self.hl_.transform(X) 
        return np.dot(self.hidden_, self.w_) 
