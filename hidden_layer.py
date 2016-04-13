from  sklearn.cluster import KMeans 
from  sklearn.metrics import pairwise_distances
from  kernels import *
import numpy as np 
from math import sqrt 

class HiddenLayer():
    
    def __init__(self, k=10, centers=None,p=1.0,compute_widths=True):
        self.k = k 
        self.centers = centers
        #        self.kernels = np.array([ create_kernel(p=p) for _ in range(self.k) ])
        self.p = p 
        self.compute_widths = compute_widths 

    def fit(self,X):
        if not self.centers:
            self._set_centers(X)
        self.kernels = np.array([ create_kernel(p=self.p) for _ in range(self.k)])
        if self.compute_widths:
            self._set_widths(X) 
        

    def _set_centers(self, X):
        print("Setting centers %s" % self.k)
        kmeans = KMeans(n_clusters=self.k)  
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

    def _set_widths_max_dist():
        max_dist = np.max(pairwise_distances(self.centers))
        for kernel in self.kernels:
            # 1/width
            kernel.set_param(sqrt(2.0 * self.k)/max_dist)

    def _set_widths_nearest_neigbor(n=3):
        nbrs = NearestNeigbors(n_neighbors=3, algorithm='ball_tree').fit(self.centers)
        for i in range(len(self.centers)):
            distances, indices = nbrs.kneighbors(self.centers[i]) 
            width = 0.1*sum(distances[0])/len(distances[0])
            self.kernels[i].set_param(1/width) 

    def _set_widths(self, X, type="nearest_neigbor"):
        print("Setting widths")
        widths_func = getattr(self, "_set_widths_"+type)  
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
