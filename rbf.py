from sklearn.base import BaseEstimator, RegressorMixin
import scipy.linalg as linalg 
import numpy as np 
from hidden_layer import *

class RBFNet(BaseEstimator,RegressorMixin): 
    """
    RBF Network 

    Simple implementation of RBF newtork, a feed-forward network with
    hidden layer of kernel units and linear output layer.
    

    Parameters:
    -----------
    k : int, optional (default=10)
        Number of hidden units, recommended to specify.  
    
    kernel : string, optional (default='Gaussian') 
             Type of kernel function to use.
             Available types: 'Gaussian', 'Multiquadric', 'InverseMultiquadric', 
                              'ProductKernel', 'SumKernel'.

    p : float, optional (default=1.0) 
        Parameter of kernel function, recommended to specify. 
    
    set_centers : string, optional (default='random')
                  The method used to set centers: 'random' or 'kmeans'. 
                  Kmeans typically perform better for small number of hidden units.

    compute_widths : string, optional (default='none') 
                     'none', 'max_dist', 'nearest_neighbor'
                     'max_dist' set all widths proportional to maximal distance 
                     between centers.  
                     'nearest_neighbor' distances are proportional to mean distance 
                      from n nearest neighbors.
    
    n_neighbors: int, optional (default=3)
                 Number of nearest neighbors if widths are calculated 
                 using nearest_neighbors.
    
    verbose : bool, optional (default=True)
    
    """
    def __init__(self, k=10, kernel="Gaussian", p=1.0, set_centers="random", compute_widths='none', verbose=True):
        self.k = k
        self.kernel = kernel 
        self.p = p
        self.compute_widths = compute_widths
        self.set_centers = set_centers
        self.verbose = verbose

    def fit(self,X,y):
        self.hl = HiddenLayer(self.k, kernel=self.kernel, p=self.p, compute_widths=self.compute_widths, 
                              set_centers=self.set_centers, verbose=self.verbose) 
        # Computes hidden layer actiovations.
        self.hidden_ = self.hl.fit_transform(X)
        # Computes output layer weights. 
        if self.verbose: print("Solving output weights.")
        self.w_ = np.dot(linalg.pinv2(self.hidden_),y) 
        return self 

        
    def predict(self,X):
        self.hidden_ = self.hl.transform(X) 
        return np.dot(self.hidden_, self.w_) 
