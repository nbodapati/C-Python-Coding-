#Given a word - get its word embedding
#From the list of glosses and their vector rep
#find those that are closest.
#see which all are closest.
import pandas as pd
import numpy as np
import re
import sys
import time
from scipy.spatial import distance

word='red'

words_=word.split()

word_embeddings=pd.read_csv('./word_embeddings_100d.csv',header=None)
word_embeddings=word_embeddings.drop(labels=[0],axis=1)
word_embeddings=word_embeddings.rename(columns={101:'words'})
#print(word_embeddings.head())
words_wnids_gloss=pd.read_csv('./words_wnids_gloss_imagenet.csv')
#print(words_wnids_gloss.head(5))
gloss_rep=pd.read_csv('./words_wnids_gloss_embeddings.csv',header=None)
gloss_rep=gloss_rep.drop(labels=[0],axis=1)
#print(gloss_rep.head())

#this allows for a string.
word_rep=np.zeros((1,100))
for word in words_:
    word_rep+=word_embeddings.loc[word_embeddings['words']==word,word_embeddings.columns!='words'].as_matrix()

neighbors=[]
for gloss_idx in range(gloss_rep.shape[0]):
    gloss_=gloss_rep.iloc[gloss_idx,:].as_matrix().reshape(*word_rep.shape)
    #print(word_rep.shape,gloss_.shape)
    cdist_=distance.cdist(word_rep,gloss_,'euclidean')
    print(cdist_)
    #if closer than 0.5 - call it a neighbor.
    if(cdist_<5):
        print("*************Potential neighbor:  ",words_wnids_gloss.loc[gloss_idx,'gloss'])
        neighbors.append(words_wnids_gloss.iloc[gloss_idx,:].to_dict())
    else:
        print("Not a neighbor:  ",words_wnids_gloss.loc[gloss_idx,'gloss'])
       
