from sklearn.base import BaseEstimator, RegressorMixin
import scipy.linalg as linalg 
from scipy.optimize import minimize
import numpy as np 
from hidden_layer import *
from sklearn.metrics import mean_squared_error

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

    def fit(self, X, y):
        self.hl = HiddenLayer(self.k, kernel=self.kernel, p=self.p, compute_widths=self.compute_widths, 
                              set_centers=self.set_centers, verbose=self.verbose) 
        # Computes hidden layer actiovations.
        self.hidden_ = self.hl.fit_transform(X)
        # Computes output layer weights. 
        if self.verbose: print("Solving output weights.")
        self.w_ = np.dot(linalg.pinv2(self.hidden_),y)
        return self 

        
    def predict(self, X):
        self.hidden_ = self.hl.transform(X) 
        return np.dot(self.hidden_, self.w_) 


    def _parameters(self):
        params = self.hl.parameters()
        params = np.hstack((params, self.w_.ravel()))    
        return params

    def _set_parameters(self, parameters):
        lenc = self.k*self.n 
        self.hl.set_parameters(parameters[:lenc+self.k])
        self.w_ = parameters[lenc+self.k:].reshape((self.k,self.m)) 


    def _objective(self, parameters):
        self._set_parameters(parameters) 
        yy = self.predict(self.X) 
        return 0.5*mean_squared_error(self.Y, yy) 

    def _derivative(self, parameters):
        deriv = np.zeros( parameters.shape )
        self._set_parameters(parameters) 
        YY = self.predict(self.X) 
        E = YY-self.Y
        HiddenDeriv_p = self.hl.deriv_p(self.X)
        HiddenDeriv_c = self.hl.deriv_c(self.X) 
        lenc = self.k * self.n
        for t in range(len(self.X)):
            for k in range(self.k): 
                for q in range(self.m):
                    #centers 
                    deriv[k*self.n:(k+1)*self.n] += E[t][q]*self.w_[k][q]*HiddenDeriv_c[t][k] 
                    # widths
                    deriv[lenc+k] += E[t][q]*self.w_[k][q]*HiddenDeriv_p[t][k]
                    # output weights 
                    deriv[lenc+self.k+k*self.m+q] += E[t][q]*self.hidden_[t][k]
        return deriv / (len(self.X)*self.m)

    def _minimize(self, X, Y):
        self.X = X
        self.Y = Y 
        self.n = X.shape[1] # input dimension 
        self.m = Y.shape[1] # output dimension 
        
        opt_params = minimize(self._objectives, self._parameters, jac = self._derivative)
        self._set_parameters(opt_params)


