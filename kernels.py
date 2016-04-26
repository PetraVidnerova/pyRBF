import math 
import numpy as np 
#from numpy.linalg import norm as norm2

__all__ = [ "BaseKernel", "Gaussian", "Multiquadric",  "InverseMultiquadric", "ProductKernel", "SumKernel", "create_kernel" ] 

def norm2(x):
    return sum(x*x)


class BaseKernel():
    """
    Base class for a kernel.
    """
    def __init__(self, p):
        self.p = p 
        
    def set_param(self, p):
        self.p = p 

    def get_param(self):
        return self.p


class Gaussian(BaseKernel):
    """
    Gaussian kernel.
    """
    def eval(self, x, y):
        return np.exp(-self.p*norm2(x-y))

    def deriv_y(self, x, y):
        # TODO 
        return np.exp(-self.p*norm2(x-y))

    def deriv_p(self, x, y):
        return -norm2(x-y)*np.exp(-self.p*norm2(x-y)) 

class Multiquadric(BaseKernel): 
    """
    Multiquadric kernel.
    """
    def eval(self,x,y):
        return np.sqrt(1.0 + self.p*norm2(x-y))

class InverseMultiquadric(BaseKernel): 
    """
    InverseMultiquadric Kernel.
    """
    def eval(self,x,y):
        return 1.0/np.sqrt(self.p*self.p+norm2(x-y))
            

class ProductKernel(BaseKernel): 
    """
    Product kernel.
    
    Parameters:
    -----------
    p : list of touples (type, p, idices) of subkernels

    """
    def __init__(self, p):
        self.sub_kernels = [] 
        self.indices_list = [] 
        for (type, p, indices) in p:
            self.sub_kernels.append(create_kernel(type, p)) 
            self.indices_list.append(indices)

    def set_param(self, p):
        # TODO: do it better, check if only p changed 
        self.sub_kernels = [] 
        self.indices_list = [] 
        for (type, p, indices) in p:
            self.sub_kernels.append(create_kernel(type, p)) 
            self.indices_list.append(indices)
                
    def eval(self, x, y):
        res = 1.0
        for kernel, indices in zip(self.sub_kernels, self.indices_list):
            res *= kernel.eval(x[indices], y[indices])
        return res 

class SumKernel(BaseKernel):
    """
    Sum kernel. 

    Parameters:
    -----------
    p : list of touples (type, p) of subkernels 
    
    """
    def __init__(self, p):
        self.sub_kernels = [] 
        for (type, p) in p:
            self.sub_kernels.append(create_kernel(type, p))

    def set_param(self, p):
        # TODO: check if change is necessary 
        self.sub_kernels = [] 
        for (type, p) in p:
            self.sub_kernels.append(create_kernel(type, p))

    def eval(self, x, y):
        res = 0 
        for kernel in self.sub_kernels:
            res += kernel.eval(x,y) 
        return res 



def create_kernel(kernel_type="Gaussian",p=1.0):
    """
    Create kernel of type given by kernel_type.
    An extra parameter of kernel is given by p.
    """
    if kernel_type == "Gaussian":
        return Gaussian(p)
    if kernel_type == "Multiquadric":
        return Multiquadric(p)
    if kernel_type == "InverseMultiquadric":
        return InverseMultiquadric(p)
    if kernel_type == "ProductKernel":
        return ProductKernel(p) 
    if kernel_type == "SumKernel":
        return SumKernel(p)
    raise Exception("No valid kernel")
