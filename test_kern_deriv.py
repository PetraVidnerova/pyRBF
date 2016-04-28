from kernels import * 
import numpy as np 

p = 0.01 
k = Gaussian(p) 

c = np.array([1, 2, 3]) 
w = np.random.random(3) 


val1 = k.eval(w, c) 
delta = 0.00001 
p = p + delta 
k.set_param(p)
val2 = k.eval(w, c) 

#deriv p
deriv = (val2-val1)/delta 
deriv2 = k.deriv_p(w, c) 
print(deriv, deriv2) 

p = 0.01
k.set_param(p)
val1 = k.eval(w, c)
delta = 0.1
deriv_c = k.deriv_c(w, c) 
#deriv c 
for i in range(len(c)):
    c2 = c.copy()
    c2[i] += delta 
    val2 = k.eval(w, c2)
    deriv = (val2-val1)/delta 
    print(deriv, deriv_c[i])
