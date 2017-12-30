from __future__ import division
import pickle
from collections import OrderedDict
import math
import nltk
import pandas as pd

KB=pickle.load(open('KnowledgeBase.pkl','rb'))
unigram_vocab=pickle.load(open('unigram_vocab.pkl','rb'))
bigram_vocab=pickle.load(open('bigram_vocab.pkl','rb'))
trigram_vocab=pickle.load(open('trigram_vocab.pkl','rb'))
bigram_pos=pickle.load(open('bigram_pos.pkl','rb'))
trigram_pos=pickle.load(open('trigram_pos.pkl','rb'))


def sort_dicts(d):
    d_=sorted(d.items(),key=lambda x: -x[1])
    return OrderedDict(d_)

def print_dict(d,name):
    print("********************************************",name)
    for k,v in d.items():
        print("Key:",k," value:",v)

def calculate_PMI(words,freq):
    #pmi(x,y)=log{p(x,y)/p(x)p(y)}
    global unigram_vocab
    words_=words.split()
    sep_freq=1
    num_elements=1#len(unigram_vocab.keys())
    for word in words_:
        sep_freq*=(unigram_vocab[word]/num_elements)

    return (freq/sep_freq) #cannot be 0.

def rectify_frequencies(d):
    new_d={}
    num_elements=1#len(d.keys())
    for words,freq in d.items():
        new_freq=calculate_PMI(words,(freq/num_elements))
        new_d[words]=new_freq 
    
    return new_d

def set_frequencies(d):
    new_d={}
    for words,freq in d.items():
        new_d[words]=1.0 
    
    return new_d


def rectify_frequencies_Pos(d,d2):
    #multiply the above rect-freq with p(tag freq from d2)
    new_d={}
    num_elements=1#len(d.keys())
    for words,freq in d.items():
        words_=words.split()
        tags=nltk.pos_tag(words_) 
        tags=[t[1] for t in tags]
        tags=' '.join(tags)
        num_d2=len(d2.keys()) 
        new_d[words]=freq*d2[tags]/num_d2  
    return new_d

def optimal_phrases():
    global bigram_vocab,trigram_vocab
    optimal=[]
    quality=[]
    num_items=len(trigram_vocab.keys())
    for key,v in trigram_vocab.items():
        print("Original phrase:",key,"To go...",num_items)
        score1=v
        words=key.split() 
        score2=bigram_vocab[' '.join(words[:-1])]  
        score3=bigram_vocab[' '.join(words[1:])]  
        scores=[score1,score2,score3]
        phrases=[key,' '.join(words[:-1]),' '.join(words[1:])]
        best_score=max(scores)
        idx=scores.index(best_score)
        optimal.append(phrases[idx])
        try:
           q=eval(input('Prompt quality(0/1): %s'%(phrases[idx]))) 
           quality.append(int(q))
           num_items-=1
        except:
           break 

    optimal_df=pd.DataFrame(dict(optimal_phrases=optimal,quality=quality)) 
    optimal_df.to_csv('./Optimal_Phrases.csv')

if __name__ =='__main__':
   #KB=sort_dicts(KB)
   #print_dict(KB,"KB")

   bigram_vocab=rectify_frequencies(bigram_vocab)
   bigram_vocab=sort_dicts(bigram_vocab)
   #print_dict(bigram_vocab,"Bigrams")

   #bigram_pos=sort_dicts(bigram_pos)
   #print_dict(bigram_pos,"Bigram_Pos")

   bigram_vocab_pos=rectify_frequencies_Pos(bigram_vocab,bigram_pos)
   bigram_vocab=sort_dicts(bigram_vocab_pos)
   print_dict(bigram_vocab,"Bigrams_Pos")

   trigram_vocab=rectify_frequencies(trigram_vocab)
   trigram_vocab=sort_dicts(trigram_vocab)
   #print_dict(trigram_vocab,"Trigrams")

   trigram_vocab_pos=rectify_frequencies_Pos(trigram_vocab,trigram_pos)
   trigram_vocab=sort_dicts(trigram_vocab_pos)
   print_dict(trigram_vocab,"Trigrams_Pos")

   #should be done at the end - the other two depend on this.
   unigram_vocab=set_frequencies(unigram_vocab)
   unigram_vocab=sort_dicts(unigram_vocab)
   #print_dict(unigram_vocab,"Unigrams")
   optimal_phrases()
