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
fourgram_vocab=pickle.load(open('fourgram_vocab.pkl','rb'))
fivegram_vocab=pickle.load(open('fivegram_vocab.pkl','rb'))
sixgram_vocab=pickle.load(open('sixgram_vocab.pkl','rb'))
#bigram_pos=pickle.load(open('bigram_pos.pkl','rb'))
#trigram_pos=pickle.load(open('trigram_pos.pkl','rb'))


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
        words=' '.join(words)
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
           if(best_score<1):
              quality.append(0)
           else:
              quality.append(1)
           #q=eval(input('Prompt quality(0/1): %s'%(phrases[idx]))) 
           #quality.append(int(q))
           num_items-=1
        except:
           break 

    optimal_df=pd.DataFrame(dict(optimal_phrases=optimal,quality=quality)) 
    optimal_df.to_csv('./Optimal_Phrases.csv')

def exists_in_KB(phrase):
    global KB
    try:
       print("Phrase:",phrase,"Score:",KB[phrase])
       return 1
    except:
       return 0 
       #import sys
       #print("Error: ",sys.exc_info()[1])  

def check_existence(d):
    print("**********************Checking existence******************")
    count=0
    for phrase,score in d.items():
         c=exists_in_KB(phrase)
         count+=c
    print("Number of exist:",count)

def lower_keys(d):
    new_d={}
    for k,v in d.items():
        new_d[k.lower()]=v
    return new_d
    

if __name__ =='__main__':
   KB=sort_dicts(KB)
   KB=lower_keys(KB)
   print_dict(KB,"KB")

   bigram_vocab=rectify_frequencies(bigram_vocab)
   #bigram_vocab_pos=rectify_frequencies_Pos(bigram_vocab,bigram_pos)
   bigram_vocab=lower_keys(bigram_vocab)
   bigram_vocab=sort_dicts(bigram_vocab)
   print_dict(bigram_vocab,"Bigrams")
   check_existence(bigram_vocab)

   trigram_vocab=rectify_frequencies(trigram_vocab)
   #trigram_vocab_pos=rectify_frequencies_Pos(trigram_vocab,trigram_pos)
   trigram_vocab=lower_keys(trigram_vocab)
   trigram_vocab=sort_dicts(trigram_vocab)
   print_dict(trigram_vocab,"Trigrams")
   check_existence(trigram_vocab)

   fourgram_vocab=rectify_frequencies(fourgram_vocab)
   fourgram_vocab=lower_keys(fourgram_vocab)
   fourgram_vocab=sort_dicts(fourgram_vocab)
   print_dict(fourgram_vocab,"Fourgrams")
   check_existence(fourgram_vocab)

   fivegram_vocab=rectify_frequencies(fivegram_vocab)
   fivegram_vocab=lower_keys(fivegram_vocab)
   fivegram_vocab=sort_dicts(fivegram_vocab)
   print_dict(fivegram_vocab,"Fivegrams")
   check_existence(fivegram_vocab)

   sixgram_vocab=rectify_frequencies(sixgram_vocab)
   sixgram_vocab=lower_keys(sixgram_vocab)
   sixgram_vocab=sort_dicts(sixgram_vocab)
   print_dict(sixgram_vocab,"Sixgrams")
   check_existence(sixgram_vocab)

   #should be done at the end - the other two depend on this.
   unigram_vocab=set_frequencies(unigram_vocab)
   unigram_vocab=lower_keys(unigram_vocab)
   unigram_vocab=sort_dicts(unigram_vocab)
   print_dict(unigram_vocab,"Unigrams")
   check_existence(unigram_vocab)
 
   #optimal_phrases()
