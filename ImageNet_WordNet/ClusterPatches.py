import pickle
from sklearn.cluster import KMeans
import time

patches=pickle.load(open('patches.pkl','rb'))

inertia={}

for n_clusters in range(2,1000,18):
    kmeans=KMeans(n_clusters=n_clusters)
    start=time.time()
    kmeans.fit(patches)
    print("Time to fit..",time.time() - start)
    inertia_=kmeans.inertia_
    print("Inertia..",inertia_)
    inertia[n_clusters]=inertia_

pickle.dump(inertia,open('cluster_evaluation.pkl','wb'))
    
