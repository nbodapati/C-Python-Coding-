import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
from Kmeans import Kmeans as Kmeans
import pickle

#read the aligned_images.csv
#each row is an image flattened to 64*64*3
print("Loading data..")
start=time.time()
images=pd.read_csv('Aligned_Images.csv',header=None)
print("Time to read..",time.time() - start)
print(images.head())

removed_images=[(images==0).all(axis=1)]
images=images.loc[~(images==0).all(axis=1)]
index=images.index
print("Remaining index..",index)
print("Images shape after removing all zeros:",images.shape)
#from dataframe to numpy matrix.
images=images.as_matrix()
kmeans=Kmeans(n_clusters=4,n_iter=1,mode='k-means')
centers,labels=kmeans.fit(images)

images_df=pd.DataFrame(images)
images_df['labels']=labels
images_df['index']=index
images_df.to_csv('Clustered_images.csv')

#cluster the images using k-means.
#visualize using bokeh. 
#do expression normalization using optical flow with 
#a reconstruction from eigen faces.
