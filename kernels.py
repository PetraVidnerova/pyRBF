import math 
import numpy as np 
#from numpy.linalg import norm as norm2

__all__ = [ "Gaussian", "Multiquadric",  "InverseMultiquadric", "create_kernel"] 

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


class Gaussian(BaseKernel):
    """
    Gaussian kernel.
    """
    def eval(self,x,y):
        return np.exp(-self.p*norm2(x-y))


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
    raise Exception("No valid kernel")
