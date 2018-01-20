#Implement k-means in own way
#define it as a class
#repeat n number of times.
#n_clusters already known. pick a random point - the second the most distant from it.
#and so on. 
#implement the algorithm. 'k-means' or 'k-medians'

import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import time
import pickle


class Kmeans(object):
      def __init__(self,n_clusters=4,n_iter=1,mode='k-means'):
          self.n_clusters=n_clusters
          self.n_iter=n_iter
          self.mode=mode

      def getNextPt(self,selected_centers,X):
          #pick a point at highest distance from all points in selected.
          max_dist=0
          cdist_=cdist(X,selected_centers) #shape (X.shape[0],selected_centers.shape[0])
          nextPt=np.argmax(np.sum(cdist_,axis=1)) 
          return nextPt       

      def pick_centers(self,X):
          centers=np.zeros((self.n_clusters,X.shape[1]))
          selected=[]
          firstPt=np.random.randint(X.shape[0],size=1)[0]
          selected.append(firstPt)      
          centers[0,:]=X[firstPt,:]
          selected_centers=X[firstPt,:].reshape(1,-1)

          for i in range(1,self.n_clusters):
              nextPt=self.getNextPt(selected_centers,X) 
              if(nextPt not in selected):
                 centers[i,:]=X[nextPt,:]
                 selected_centers=np.append(selected_centers,X[nextPt,:].reshape(1,-1),axis=0)
                 selected.append(nextPt)

          print("Selected centers/..",selected)
          return centers

      def getNewCenters(self,centers,X):
          #assign points to centers and use mode to decide new centers.
          #labels is the argmin index to the centers. 
          ptsCdist=cdist(X,centers)
          labels=np.argmin(ptsCdist,axis=1)

          for i in range(self.n_clusters):          
              if(self.mode=='k-means'):
                 centers[i]=np.mean(X[labels==i,:])
              elif(self.mode=='k-medians'):
                 centers[i]=np.median(X[labels==i,:])

          return centers,labels       
              
      def fit_iter(self,X):
          #unsupervised X - shape NxD
          start=time.time()
          centers=self.pick_centers(X)      
          print("Time to fit..",time.time() - start)
          
          labels=np.zeros((1,X.shape[0])) #initially all belong to same cluster.
          old_centers=centers.copy()
          centers,labels=self.getNewCenters(centers,X)   
          dist_centers=np.sum(np.sqrt(np.sum((old_centers-centers)**2,axis=1))) 
          print("Distance between centers..",dist_centers)
          
          iter_num=1
          while(dist_centers):
               centers,labels=self.getNewCenters(centers,X)   
               dist_centers=np.sum(np.sqrt(np.sum((old_centers-centers)**2,axis=1))) 
               old_centers=centers.copy()
               print("Distance between centers..",dist_centers)
               print("Iter num..",iter_num)
               iter_num+=1 
          return centers,labels
 
      def fit(self,X):
          for i in range(self.n_iter):
              centers,labels=self.fit_iter(X)   

          intertia_=self.inertia(centers,labels)
          return centers,labels,inertia_

      def inertia(self,centers,labels):
          inertia_=0
          #sum of distances of all points in a cluster to its center.
          for i in range(self.n_clusters):          
                inertia_+=np.sum(cdist(X[labels==i,:],centers[i,:]))

          print("Inertia of clusters..",inertia_) 
          return inertia_
