from  sklearn.cluster import KMeans 
from  sklearn.metrics import pairwise_distances
from  kernels import *
import numpy as np 
from math import sqrt 
from sklearn.neighbors import NearestNeighbors 

class HiddenLayer():
    
    def __init__(self, k=10, kernel="Gaussian", centers=None,p=1.0,compute_widths='none',n_neighbors=3, set_centers="random", verbose=True):
        self.k = k 
        self.centers = centers
        self.kernel = kernel 
        self.p = p 
        self.compute_widths = compute_widths 
        self.set_centers = set_centers
        self.n_neighbors = n_neighbors 
        self.verbose = verbose

    def fit(self,X):
        self.n = X.shape[1] 
        if not self.centers:
            self._set_centers(X)
        #create the array of kernels 
        self.kernels = np.array([ create_kernel(kernel_type=self.kernel, p=self.p) for _ in range(self.k)])
        self._set_widths(self.compute_widths) 

        
    def _set_centers_kmeans(self, X):
        kmeans = KMeans(n_clusters=self.k)  
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

    def _set_centers_random(self, X):
        random_indices = np.random.choice(range(len(X)), self.k, replace=False)
        self.centers = X[random_indices]

    
    def _set_centers(self, X):
        if self.verbose: print("Setting centers %s %s" % (self.k, self.set_centers))
        centers_func = getattr(self, "_set_centers_"+self.set_centers)
        centers_func(X)
        

    def _set_widths_max_dist(self):
        max_dist = np.max(pairwise_distances(self.centers))
        for kernel in self.kernels:
            # 1/width
            kernel.set_param(self.p*sqrt(2.0 * self.k)/max_dist)

    def _set_widths_nearest_neighbor(self):
        # Nearest neighbors contain center itself, find one more.
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors+1, algorithm='ball_tree').fit(self.centers)
        for i in range(len(self.centers)):
            distances, indices = nbrs.kneighbors(self.centers[i]) 
            width = sum(distances[0])/(len(distances[0]-1))
            self.kernels[i].set_param(self.p/width) 

    def _set_widths(self, compute_widths="none"):
        if compute_widths == "none":
            return 
        if self.verbose: print("Setting widths")
        widths_func = getattr(self, "_set_widths_"+compute_widths)  
        widths_func() 
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X) 
        
    def transform(self, X, y=None): 
        self.hidden_ = np.zeros((X.shape[0],self.k))
        for i in range(X.shape[0]):
            for j in range(self.k):
                self.hidden_[i][j] = self.kernels[j].eval(X[i],self.centers[j])
        return self.hidden_

    def deriv_p(self, X):
        deriv = np.zeros((len(X), self.k))
        for i in range(len(X)):
            for j in range(self.k):
                deriv[i][j] = self.kernels[j].deriv_p(X[i], self.centers[j]) 
        return deriv 
        
    def deriv_c(self, X):
        deriv = np.zeros((len(X), self.k, self.n))
        for i in range(len(X)):
            for j in range(self.k):
                deriv[i][j] = self.kernels[j].deriv_c(X[i], self.centers[j])
        return deriv

    def parameters(self):
        params = self.centers.ravel() 
        lenc = self.k*self.n 
        params = np.hstack((params, np.zeros(self.k))) 
        for k in range(self.k):
            params[lenc+k] = self.kernels[k].get_param()  
        return params

        
    def set_parameters(self, params):
        lenc = self.k*self.n 
        self.centers = params[:lenc].reshape(self.k, self.n)
        for k in range(self.k):
            self.kernels[k].set_param(params[lenc+k]) 
            

        

