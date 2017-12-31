from __future__ import division
import pandas as pd
import numpy as np
from collections import defaultdict,OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
import pickle

df=pd.read_csv('./Optimal_Phrases.csv')

pos_df=df[df['quality']==1]
neg_df=df[df['quality']==0]
neg_df=neg_df.iloc[::10,:]
print(pos_df.shape,neg_df.shape)

df_=pd.concat([pos_df,neg_df])
phrases=df_['optimal_phrases']
labels=df_['quality']
vocab=defaultdict(int)

def build_vocab():
    global phrases,vocab
    for phrase in phrases:
        words=phrase.split()
        for word in words:
            vocab[word]+=1

def prune_vocab():
    #feature selection in nlp.
    #proportional to term frequency.
    new_vocab=defaultdict(int)
    for w,f in vocab.items():
        if(f>10):
            new_vocab[w]=f

    print("Len new_vocab:",len(new_vocab))
    return new_vocab

def top_n_words(vocab,n=1000):
    sorted_=sorted(vocab.items(),key=lambda x: -x[1])
    words,_=zip(*sorted_)
    return words[:n]

def build_features(new_vocab):
    global phrases
    features_df=pd.DataFrame(np.zeros((len(phrases),len(new_vocab))),columns=new_vocab)
    for i,phrase in enumerate(phrases):
        words=phrase.split()
        for word in words:
            features_df.loc[i,word]+=1

    return features_df.as_matrix()

def print_vocab(vocab):
    i=0
    for word,f in vocab.items():
        print(word,f,end='  ')
        i=(i+1)%5
        if(i==4):
            print()

def print_list(lst):
    for l in lst:
        print(l,end=' ')
    print()

def train_ensemble(X_train,y_train):
    n_trees=5
    #should be as diverse and good in accuracy as possible.
    #list of decision trees - each trained to unpruned depth.
    dtrees=[DTC(criterion='entropy',max_features='log2') for _ in range(n_trees)]
    trained_trees=[]
    for dt in dtrees:
        print(dt)
        dt.fit(X_train,y_train) 
        trained_trees.append(dt)

    return trained_trees

def predict_tree(tree,X_test,Y_test):
    #tree - trained tree.
    pred=tree.predict(X_test) 
    acc=np.mean(pred==Y_test)
    print("Accuracy: ",acc)
    return pred

def combined_accuracy(preds,y_test):
    global vocab
    preds=np.array(preds).reshape(-1,5)
    print("Preds shape: ",preds.shape)
    #shape - NxT
    fpred=np.mean(preds,axis=0,keepdims=True)
    print(fpred.shape)
    preds_df=pd.DataFrame(preds)
    preds_df['y_test']=y_test
    #preds_df['fpred']=fpred

    return (preds_df,0)

def map_vector2sentence(vector,new_vocab):
    #vector - sparse matrix with 1s at locations of words
    #new_vocab - list of words that are used in building the features.
    new_vocab=np.array(new_vocab)
    vector=(vector>0).tolist() 
    #print(vector,new_vocab)
    selected_words=new_vocab[vector]
    sentence=' '.join(selected_words)
    print(sentence)

def get_testsentences(x_test,new_vocab):
    sentences=[]
    for vector in range(x_test.shape[0]):
        vec=x_test[vector,:]
        sentences.append(map_vector2sentence(vec,new_vocab))
    return sentences

if __name__ =='__main__':
   build_vocab()
   print(len(vocab))
   #print_vocab(vocab)
   new_vocab=top_n_words(vocab,n=len(vocab))#prune_vocab()
   #print(new_vocab)
   features=build_features(new_vocab)
   print_list(np.sum(features,axis=1))

   X_train, X_test, y_train, y_test = train_test_split(\
      features, labels, test_size=0.45, random_state=42)
   print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
   
   trained_trees=train_ensemble(X_train,y_train)
   preds=[]
   for tree in trained_trees:
       preds.append(predict_tree(tree,X_test,y_test))   
   
   preds_df,cacc=combined_accuracy(preds,y_test)  
   test_df=pd.DataFrame(dict(sentences=get_testsentences(X_test,new_vocab),y_test=y_test))
   #preds_df=pd.concat([preds_df,test_df])
   test_df.to_csv('./TestSet.csv')
   preds_df.to_csv('./Predictions_Ensemble.csv')
   pickle.dump(vocab,open('vocab.pkl','wb'))
   
   print("Combined accuracy: ",cacc)
