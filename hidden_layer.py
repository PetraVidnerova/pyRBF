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
        if not self.centers:
            self._set_centers(X)
        #create the array of kernels 
        
        self.kernels = np.array([ create_kernel(kernel, self.p) for _ in range(self.k)])

        self._set_widths(self.compute_widths) 

        def create_kernel(kernel="Gaussian",p=1.0):
            """
            Create kernel of type given by kernel_type.
            An extra parameter of kernel is given by p.
            """
            if kernel == "Gaussian":
                return Gaussian(p)
            if kernel == "Multiquadric":
                return Multiquadric(p)
            if kernel == "InverseMultiquadric":
                return InverseMultiquadric(p)
            if kernel == "ProductKernel":
                return ProductKernel(p) 
            if kernel == "SumKernel":
                return SumKernel(p) 
            raise Exception("No valid kernel")

        
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
