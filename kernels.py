import math 
import numpy as np 

__all__ = [ "Gaussian" ] 

def norm2(vec):
    return sum(vec*vec) 

class Gaussian():
    
    def __init__(self, p): 
        self.p = p 

    def eval(self,x,y,w):
        return math.exp(-self.p*norm2(x-y)/w)


 
