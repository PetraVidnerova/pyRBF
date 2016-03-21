from kernels import Gaussian 


def test():
    g = Gaussian(np.array([0,0]),0.1) 
    x = -5.0
    while (x<5.0):
        y = -5.0 
        while (y<5.0):
            print(x,y,g.eval(np.array([x,y]))) 
            y += 0.1 
        x += 0.1 

test() 
