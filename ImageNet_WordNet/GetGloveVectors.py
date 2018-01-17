from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import requests
import random
import sys
import re

word_embeddings=pd.read_csv('./word_embeddings_100d.csv',header=None)
word_embeddings=word_embeddings.drop(labels=[0],axis=1)
word_embeddings=word_embeddings.rename(columns={101:'words'})

wnids_pc=pd.read_csv('./wnids_parent_child.csv')
wnids_pc=wnids_pc.drop(labels=['Unnamed: 0'],axis=1)
x=pd.read_csv('./words_wnids_gloss_imagenet.csv')
print(x.shape)
xml_dataframe=x.dropna(how='any')
xml_dataframe=xml_dataframe.drop(labels=['Unnamed: 0'],axis=1)
print(wnids_pc.shape,xml_dataframe.shape)
wnids_pc.head(),xml_dataframe.head()

glosses=xml_dataframe['gloss'].tolist()
glosses_arr=np.zeros((1,100))


def preprocess(text):
    text=re.sub(r'[^a-zA-Z0-9\s]',' ',text)
    tokens= text.split()
    return [tok for tok in tokens if len(tok)>=3]

def sumup(tokens):
    global word_embeddings
    gloss_rep=np.zeros((1,100))
    for tok in tokens:
        try:
            print(tok,file=file_out)
            rep=word_embeddings.loc[word_embeddings['words']==tok,word_embeddings.columns!='words'].as_matrix()
            if(len(rep)==0):
               rep=word_embeddings.loc[word_embeddings['words']==tok.lower(),word_embeddings.columns!='words'].as_matrix()  
               if(len(rep)==0):
                 continue
            gloss_rep=gloss_rep+rep
            print(tok,rep,file=file_out)
        except:
            print(sys.exc_info()[1],file=file_out)
            print("Missing tok.. ",tok,file=file_out)

    return gloss_rep/len(tokens) #normalize this to nullify effect of number of tokens in the string.
i=0
for gloss_ in glosses:
    file_out=open('gloss_conversion.txt','a')
    print(i,gloss_,file=file_out)
    i+=1
    tokens=preprocess(gloss_)
    print(tokens,file=file_out)
    rep=sumup(tokens)
    glosses_arr=np.append(glosses_arr,rep,axis=0)
    if(i%10==0):
        full_df=pd.DataFrame(glosses_arr[1:,:])
        print(i,full_df.shape,"Written to csv..")
        glosses_arr=np.zeros((1,100))
        full_df.to_csv('./words_wnids_gloss_embeddings.csv',mode='a',header=False)
    file_out.close()

