from random import random

#generates a vector of c*2 random points and labels.
def generateData(c):
    xa = []
    ya = []
    xb = []
    yb = []
    for i in range(c):
        xa.append((random()*2-1)/2-0.5)
        ya.append((random()*2-1)/2+0.5)
        xb.append((random()*2-1)/2+0.5)
        yb.append((random()*2-1)/2-0.5)
        
    data = []
    for i in range(len(xb)):
        data.append([xa[i],ya[i],1])
        data.append([xb[i],yb[i],-1])
    return data
