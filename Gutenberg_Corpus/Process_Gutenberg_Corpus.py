import os
import gzip
import re
import nltk 
from collections import defaultdict
import pickle

path_to_files='./text/'
list_of_files=os.listdir(path_to_files)
list_of_files=sorted(list_of_files)

verb_pattern=r'VB.*'
noun_pattern=r'NN.*'
adj_pattern=r'JJ.*'
adv_pattern=r'RB.*'

vocab_dict_1=defaultdict(int)
pos_dict_1=defaultdict(int)
vocab_dict_2=defaultdict(int)
pos_dict_2=defaultdict(int)
vocab_dict_3=defaultdict(int)
pos_dict_3=defaultdict(int)
vocab_dict_4=defaultdict(int)
pos_dict_4=defaultdict(int)
vocab_dict_5=defaultdict(int)
pos_dict_5=defaultdict(int)
vocab_dict_6=defaultdict(int)
pos_dict_6=defaultdict(int)

#file_0=list_of_files[-100]
def get_text(filename):
    with gzip.open(os.path.join(path_to_files,filename),'rb') as fp:
         text=fp.read().decode('utf-8')
    fp.close()
    return text

     
def sentence_generator(text):
    sentences=text.split('\n')
    for sentence in sentences:
        if(sentence!='\r'): #empty line
            yield sentence

def preprocess_sentence(sentence):
    sentence=re.sub(r'[\']','',sentence)
    sentence=re.sub(r'[^A-Za-z\s]',' ',sentence)
    sentence=re.sub(r'[\' \']+',' ',sentence)
    return sentence

def tokenize(sentence):
    return sentence.split()

def pos_tagger(tokens):
    #return list of tuples of form (word,tag)
    return nltk.pos_tag(tokens)

def filter_tags(tokens_tags):
    filtered=[]
    for token,tag in tokens_tags:
        if(re.findall(noun_pattern,tag) or re.findall(verb_pattern,tag) or re.findall(adj_pattern,tag)):
           filtered.append((token,tag))          
    return filtered

def unigram_dict(tokens_tags):
    global vocab_dict_1,pos_dict_1
    for token,tag in tokens_tags:
        vocab_dict_1[token]+=1
        pos_dict_1[tag]+=1
 
def bigram_dict(tokens_tags):
    global vocab_dict_2,pos_dict_2

    tokens,tags=list(zip(*tokens_tags))
    for i in range(0,len(tokens)-1):
        w1=tokens[i]
        w2=tokens[i+1]   
        vocab_dict_2[(w1,w2)]+=1

    for i in range(0,len(tags)-1):
        w1=tags[i]
        w2=tags[i+1]   
        pos_dict_2[(w1,w2)]+=1

def trigram_dict(tokens_tags):
    global vocab_dict_3,pos_dict_3
    tokens,tags=list(zip(*tokens_tags))
    for i in range(0,len(tokens)-2):
        w1=tokens[i]
        w2=tokens[i+1]   
        w3=tokens[i+2]  
        vocab_dict_3[(w1,w2,w3)]+=1

    for i in range(0,len(tags)-2):
        w1=tags[i]
        w2=tags[i+1]   
        w3=tags[i+2]  
        pos_dict_3[(w1,w2,w3)]+=1

def fourgram_dict(tokens_tags):
    global vocab_dict_4,pos_dict_4
    tokens,tags=list(zip(*tokens_tags))
    for i in range(0,len(tokens)-2):
        w1=tokens[i]
        w2=tokens[i+1]   
        w3=tokens[i+2]
        w4=tokens[i+3]  
        vocab_dict_4[(w1,w2,w3,w4)]+=1

    for i in range(0,len(tags)-2):
        w1=tags[i]
        w2=tags[i+1]   
        w3=tags[i+2]
        w4=tags[i+3]  
        pos_dict_4[(w1,w2,w3,w4)]+=1

def fivegram_dict(tokens_tags):
    global vocab_dict_5,pos_dict_5
    tokens,tags=list(zip(*tokens_tags))
    for i in range(0,len(tokens)-2):
        w1=tokens[i]
        w2=tokens[i+1]   
        w3=tokens[i+2]
        w4=tokens[i+3]  
        w5=tokens[i+4]  
        vocab_dict_5[(w1,w2,w3,w4,w5)]+=1

    for i in range(0,len(tags)-2):
        w1=tags[i]
        w2=tags[i+1]   
        w3=tags[i+2]
        w4=tags[i+3]  
        w5=tags[i+4]  
        pos_dict_5[(w1,w2,w3,w4,w5)]+=1

def sixgram_dict(tokens_tags):
    global vocab_dict_6,pos_dict_6
    tokens,tags=list(zip(*tokens_tags))
    for i in range(0,len(tokens)-2):
        w1=tokens[i]
        w2=tokens[i+1]   
        w3=tokens[i+2]
        w4=tokens[i+3]  
        w5=tokens[i+4]  
        w6=tokens[i+5]  
        vocab_dict_6[(w1,w2,w3,w4,w5,w6)]+=1

    for i in range(0,len(tags)-2):
        w1=tags[i]
        w2=tags[i+1]   
        w3=tags[i+2]
        w4=tags[i+3]  
        w5=tags[i+4]  
        w6=tags[i+5]  
        pos_dict_6[(w1,w2,w3,w4,w5,w6)]+=1

def pickle_dicts():
    global vocab_dict_1,vocab_dict_2,vocab_dict_3
    global pos_dict_1,pos_dict_2,pos_dict_3

    print("Pickling dicts..")
    pickle.dump(vocab_dict_1,open('unigram_vocab.pkl','wb'))    
    pickle.dump(vocab_dict_2,open('bigram_vocab.pkl','wb'))    
    pickle.dump(vocab_dict_3,open('trigram_vocab.pkl','wb'))    
    pickle.dump(vocab_dict_4,open('fourgram_vocab.pkl','wb'))    
    pickle.dump(vocab_dict_5,open('fivegram_vocab.pkl','wb'))    
    pickle.dump(vocab_dict_6,open('sixgram_vocab.pkl','wb'))    

    pickle.dump(pos_dict_1,open('unigram_pos.pkl','wb'))    
    pickle.dump(pos_dict_2,open('bigram_pos.pkl','wb'))    
    pickle.dump(pos_dict_3,open('trigram_pos.pkl','wb'))    
    pickle.dump(pos_dict_4,open('fourgram_pos.pkl','wb'))    
    pickle.dump(pos_dict_5,open('fivegram_pos.pkl','wb'))    
    pickle.dump(pos_dict_6,open('sixgram_pos.pkl','wb'))    
    print("done with pickling..")
    return 

def build_dictionaries(tokens_tags):
    unigram_dict(tokens_tags)
    bigram_dict(tokens_tags)
    trigram_dict(tokens_tags)
    try:
      fourgram_dict(tokens_tags)
      fivegram_dict(tokens_tags)
      sixgram_dict(tokens_tags)
    except:
      pass
    return

if __name__ =='__main__':
   for i,file_ in enumerate(list_of_files):
        text=get_text(file_)
        print(i,file_,len(list_of_files)-i)
        for s in sentence_generator(text):
            s=preprocess_sentence(s)
            tokens=tokenize(s)
            tokens_tags=pos_tagger(tokens)
            tokens_tags=filter_tags(tokens_tags)
            if(tokens_tags==[]):
               continue
            else:
               pass
               #print(tokens_tags)  
            build_dictionaries(tokens_tags)
        pickle_dicts() 
 
