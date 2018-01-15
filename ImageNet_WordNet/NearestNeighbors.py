#Given a word - get its word embedding
#From the list of glosses and their vector rep
#find those that are closest.
#see which all are closest.
import pandas as pd
import numpy as np
import re
import sys
import time
from scipy.spatial.distance import cdist

word='plant'
word_embeddings=pd.read_csv('./word_embeddings_100d.csv',header=None)
word_embeddings=word_embeddings.drop(labels=[0],axis=1)
word_embeddings=word_embeddings.rename(columns={101:'words'})

gloss_rep=pd.read_csv('./words_wnids_gloss_embeddings.csv',header=None)
print(gloss_rep.head())
word_rep=word_embeddings.loc[word_embeddings['words']==word,df.columns!='words'].as_matrix()

neighbors=[]
for gloss_idx in range(gloss_rep.shape[0]):
    gloss_=gloss_rep.iloc[gloss_idx,100:].as_matrix()
    cdist_=cdist(word_rep,gloss_)
    print(cdist)
    #if closer than 0.5 - call it a neighbor.
    if(cdist<=0.5):
        print("Potential neighbor:  ",gloss_rep.iloc[gloss_idx,'words'])
        neighbors.append(gloss_rep.iloc[gloss_idx,:3].to_dict())
