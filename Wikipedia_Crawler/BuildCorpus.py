import pandas as pd
import numpy as np
import pickle
import re
import os
import csv
from collections import defaultdict,Counter
import time,sys
from nltk.corpus import stopwords
import nltk

dataDirectory=None
stopwords = stopwords.words('english')
num_files=0

#this builds three dicts with 1 word,2 words,3 word scopes.
vocab_dict_1=defaultdict(int)
vocab_dict_2=defaultdict(int)
vocab_dict_3=defaultdict(int)

def unigram_dict(sentence):
    #remove all stopwords,build vocab_dict_1
    global vocab_dict_1
    tokens_dict=Counter(sentence.split())
    for token,count in tokens_dict.items():
        if(token not in stopwords):
           vocab_dict_1[token]+=count
 
def bigram_dict(sentence):
    #remove all stopwords,build vocab_dict_1
    global vocab_dict_2
    tokens=sentence.split()
    tokens=[t for t in tokens if t not in stopwords]
    if(len(tokens)<2):
      return
    for i in range(0,len(tokens)-1):
        w1=tokens[i]
        w2=tokens[i+1]   
        vocab_dict_2[(w1,w2)]+=1

def trigram_dict(sentence):
    #remove all stopwords,build vocab_dict_1
    global vocab_dict_3
    tokens=sentence.split()
    tokens=[t for t in tokens if t not in stopwords]
    if(len(tokens)<3):
      return
    for i in range(0,len(tokens)-2):
        w1=tokens[i]
        w2=tokens[i+1]   
        w3=tokens[i+2]  
        vocab_dict_2[(w1,w2,w3)]+=1

def pickle_dicts():
    global vocab_dict_1,vocab_dict_2,vocab_dict_3
    print("Pickling dicts..")
    pickle.dump(vocab_dict_1,open('unigram_vocab.pkl','wb'))    
    pickle.dump(vocab_dict_2,open('bigram_vocab.pkl','wb'))    
    pickle.dump(vocab_dict_3,open('trigram_vocab.pkl','wb'))    
    print("done with pickling..")
    return 

def build_dictionaries():
    global num_files,dataDirectory
    for file in os.listdir(dataDirectory):
        if(os.name =="nt"): 
           filename = os.fsdecode(file)
        elif(os.name=="posix"):
            filename=file  
        num_files+=1
        print("Num file: ",num_files)
        with open(os.path.join(dataDirectory, filename), 'r') as readFile:
             document=readFile.read()
             sentences=nltk.sent_tokenize(document)
             for sentence in sentences:
                 sentence=re.sub(r'[^a-zA-Z]'," ",sentence)
                 sentence=re.sub(r'[" "]+'," ",sentence)
                 print(sentence)
                 #build the dicts
                 unigram_dict(sentence)
                 bigram_dict(sentence)
                 trigram_dict(sentence)
    return
    
def main():
    global dataDirectory
    #windows
    if(os.name =="nt"): 
       dataDirectory = os.getcwd() + "\data\\"
       dataDirectory += "nlp\\"
    #linux-like machines
    elif(os.name=="posix"):
       dataDirectory = os.getcwd() + "/data/"
       dataDirectory += "nlp/"
       reload(sys)
       sys.setdefaultencoding('latin-1')
    else:
         print("Unknown OS")

    
if __name__ == '__main__':
    main()
    build_dictionaries()
    pickle_dicts()
