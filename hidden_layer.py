from  sklearn.cluster import KMeans 
from  sklearn.metrics import pairwise_distances
from  kernels import Gaussian 
import numpy as np 
from math import sqrt 

class HiddenLayer():
    
    def __init__(self, k, centers=None,widths=None,p=1.0):
        self.k = k 
        self.centers = centers
        self.widths = widths
        self.kernel = Gaussian(p)
        pass

    def fit(self,X):
        if not self.centers:
            self.set_centers(X)
        if not self.widths:
            self.set_widths(X) 
                  
    def set_centers(self, X):
        kmeans = KMeans(n_clusters=self.k)  
        kmeans.fit(X)
        print("kmeans:")
        print(kmeans.cluster_centers_)
        self.centers = kmeans.cluster_centers_

    def set_widths(self, X): 
        max_dist = np.max(pairwise_distances(self.centers))
        self.widths = np.ones(self.k) * max_dist/sqrt(2.0 * self.k)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X) 
        
    def transform(self, X, y=None): 
        print(X.iloc[[0]])
        print(self.centers)
        print(self.centers[0]) 

        print(self.widths[0])
        self.hidden_ = [ Gaussian.eval(X.iloc[[i]],self.centers[i],self.widths[i]) for i in range(self.k) ] 
        return 
