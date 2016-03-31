from  sklearn.cluster import KMeans 
from  sklearn.metrics import pairwise_distances
from  kernels import *
import numpy as np 
from math import sqrt 

class HiddenLayer():
    
    def __init__(self, k, centers=None,p=1.0,compute_widths=True):
        self.k = k 
        self.centers = centers
        self.kernels = np.array([ create_kernel(p=p) for _ in range(k) ])
        self.compute_widths = compute_widths 

    def fit(self,X):
        if not self.centers:
            self.set_centers(X)
        if self.compute_widths:
            self.set_widths(X) 
                  
    def set_centers(self, X):
        kmeans = KMeans(n_clusters=self.k)  
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

    def set_widths(self, X): 
        max_dist = np.max(pairwise_distances(self.centers))
        for kernel in self.kernels:
            kernel.set_param(max_dist/sqrt(2.0 * self.k))

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X) 
        
    def transform(self, X, y=None): 
        self.hidden_ = np.zeros((X.shape[0],self.k))
        for i in range(X.shape[0]):
            for j in range(self.k):
                self.hidden_[i][j] = self.kernels[j].eval(X[i],self.centers[j])
        return self.hidden_
