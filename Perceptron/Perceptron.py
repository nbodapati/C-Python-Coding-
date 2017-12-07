import rand_data
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def make_plot(ai,li,weight):
    #w0*x+w1*y+w2=0;
    #slope=-w0/w1
    #int=-w2/w1

    w0=weight[0]
    w1=weight[1]
    w2=weight[2]

    slope=a=-w0/w1
    intercept=b=-w2/w1
    x = np.linspace(-5, 5)
    y = a * x + b

    points=ai[:,:2] #only the first two indices.

    plt.scatter(points[:,0],points[:,1],c=li)
    plt.plot(x, y, color='green', linestyle='dashed', marker='o',
       markerfacecolor='blue', markersize=2)
    plt.title("Decision boundatry for n={0}".format(len(li)))
    plt.show()

def calculate_margin(n,ai,li,weight):
    norm_w=LA.norm(weight)
    n_weight=weight/norm_w
    min=100
    margin=0
    for i in range(n):
        distance=np.dot((ai[i,:]*li[i]),n_weight)
        if(distance<min):
           margin=distance
           min=margin
    return margin

#sample sizes of n=4,10,20
for n in [4,10,20]:
    print("Num data points: ",n)
    data=rand_data.generateData(n//2)
    li=[]
    ai=np.ones((n,3))
    for i,dp in enumerate(data):
        ai[i,:2]=dp[:2]
        #normalize
        norm_a=LA.norm(ai[i,:])
        ai[i,:]=ai[i,:]/norm_a
        li.append(dp[2])

    weight=ai[0,:]*li[0]
    print(weight)
    loop=True
    epochs=0
    num_updates=0
    make_plot(ai,li,weight)
    while(loop):
        epochs+=1
        count=0
        for i in range(n):
            if((np.dot(weight,ai[i,:])*li[i])<0):
                count+=1 #one point is misbehaving.
                num_updates+=1
                weight+=ai[i,:]*li[i]
                print(i,weight)
                make_plot(ai,li,weight)
            if(count==0):
               loop=False #break out of the loop.
    print("Num epochs: ",epochs)
    print("Num updates: ",num_updates)
    margin=calculate_margin(n,ai,li,weight)
    print("Margin: ",margin)
