from kernels import * 
import numpy as np

def test():
    g = create_kernel("Gaussian",0.5) 
    x = -5.0
    while (x<5.0):
        y = -5.0 
        while (y<5.0):
            print(x,y,g.eval(np.array([0,0]),np.array([x,y]))) 
            y += 0.1 
        x += 0.1 

test() 
