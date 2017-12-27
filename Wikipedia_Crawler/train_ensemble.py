import pickle
from collections import OrderedDict

#KB=pickle.load(open('KnowledgeBase.pkl','rb'))
unigram_vocab=pickle.load(open('unigram_vocab.pkl','rb'))
bigram_vocab=pickle.load(open('bigram_vocab.pkl','rb'))
trigram_vocab=pickle.load(open('trigram_vocab.pkl','rb'))


def sort_dicts(d):
    d_=sorted(d.items(),key=lambda x: -x[1])
    return OrderedDict(d_)

def print_dict(d,name):
    print("********************************************",name)
    for k,v in d.items():
        print("Key:",k," value:",v)

if __name__ =='__main__':
   unigram_vocab=sort_dicts(unigram_vocab)
   print_dict(unigram_vocab,"Unigrams")
   bigram_vocab=sort_dicts(bigram_vocab)
   print_dict(bigram_vocab,"Bigrams")
   trigram_vocab=sort_dicts(trigram_vocab)
   print_dict(trigram_vocab,"Trigrams")
   #KB=sort_dicts(KB)
   #print_dict(KB,"KB")

