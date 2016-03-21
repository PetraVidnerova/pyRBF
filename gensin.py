import math 

for x in [0.1*x for x in range(-5,6,1)]:
    for y in [0.1*x for x in range(-5,6,1)]:
        print(x,y,math.sin(x),sep=";")
