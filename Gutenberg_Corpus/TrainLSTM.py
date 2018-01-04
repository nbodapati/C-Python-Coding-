from __future__ import division
import pandas as pd
import numpy as np
from collections import defaultdict,OrderedDict
from sklearn.model_selection import train_test_split
import pickle
from keras.preprocessing import sequence
import tensorflow as tf
from tensorflow.contrib import rnn
import os
from sklearn.utils import shuffle
learning_rate = 0.0001
training_epochs = 101#10000
display_step = 5
batch_size=100

time_steps=3 #trigrams
num_units=32
n_classes=2
n_input=1
df=pd.read_csv('./Optimal_Phrases.csv')
df=df.dropna(axis=0,how='any').reset_index()
print(df.shape)
pos_df=df[df['quality']==1]
neg_df=df[df['quality']==0]
#neg_df=neg_df.iloc[::6,:]
#pos_df=pos_df.iloc[::5,:]
neg_df=neg_df.iloc[::6,:]

#how to align the labels.
'''
df_=pd.concat([pos_df.iloc[0,:],neg_df.iloc[0,:]])
for i in range(1,pos_df.shape[0]):
    df_=pd.concat([df_,pos_df.iloc[i,:],neg_df.iloc[i,:]])
    
df_=pd.concat([df_,neg_df.iloc[i:,:]])
'''
n_pos=pos_df.shape[0]
n_neg=neg_df.shape[0]
df_=pd.concat([pos_df.iloc[:n_pos//2,:],neg_df.iloc[:n_neg//2,:]])
df_=pd.concat([df_,pos_df.iloc[n_pos//2:,:],neg_df.iloc[n_neg//2:,:]])
df_=df_.dropna(axis=0,how='any').reset_index()

print(pos_df.shape,neg_df.shape,df_.shape)

phrases=df_['optimal_phrases']
labels=df_['quality'].as_matrix()
vocab=defaultdict(int)
rev_vocab=defaultdict(str)

vocabulary_size=None
embedding_size=128

def append_phrases():
    global phrases
    new_phrases=[]
    for phrase in phrases: 
        new_phrases.append('<s>'+phrase+'</s>')
    phrases=new_phrases
    df_['optimal_phrases']=new_phrases

def build_vocab():
    global phrases,vocab,vocabulary_size
    for phrase in phrases:
        words=phrase.split()
        for word in words:
            vocab[word]=len(vocab.keys())
    vocabulary_size=len(vocab.keys())
    print("Vocab size: ",vocabulary_size)

def build_revvocab():
    global vocab,rev_vocab
    for k,v in vocab.items():
        rev_vocab[v]=k

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

def build_features(vocab):
    global phrases
    features=[[] for _ in range(len(phrases))]
    for i,phrase in enumerate(phrases):
        words=phrase.split()
        #if(len(words)>1): 
        #   print(words)
        for word in words:
            features[i].append(vocab[word])
        '''
        if(len(words)>1): 
           print(features[i])
        ''' 
    features=sequence.pad_sequences(features,maxlen=time_steps,padding='post') 
    print(features.shape)
    #list of lists
    return features

def print_list(lst):
    for l in lst:
        print(l,end=' ')
    print()


def map_vector2sentence(vector,new_vocab):
    #vector - sparse matrix with 1s at locations of words
    #new_vocab - list of words that are used in building the features.
    selected_words=[]
    for v in vector:
        #print(v)
        if(v!=0):
           selected_words.append(new_vocab[v])

    sentence=' '.join(selected_words)
    print(sentence)
    return sentence

def get_testsentences(x_test,new_vocab):
    sentences=[]
    for vector in range(x_test.shape[0]):
        vec=x_test[vector,:]
        print(vec)
        sentences.append(map_vector2sentence(vec,new_vocab))
    return sentences

def print_vocab(vocab):
    i=0
    for word,f in vocab.items():
        print(word,f,end='  ')
        i=(i+1)%5
        if(i==4):
            print()
#get data and train model below.
#append_phrases() 
build_vocab()
build_revvocab()

features=build_features(vocab)
#get_embeddings()
print(len(vocab))
n=int(0.65*features.shape[0])
X_train=features[:n,:]
y_train=labels[:n]
X_test=features[n:,:]
y_test=labels[n:]

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
n_samples=X_train.shape[0]
n_test=X_test.shape[0]

train_df=pd.DataFrame(dict(sentences=get_testsentences(X_train,rev_vocab),y=y_train.tolist()))
train_df.to_csv('./TrainSet.csv')

test_df=pd.DataFrame(dict(sentences=get_testsentences(X_test,rev_vocab),y=y_test.tolist()))
test_df.to_csv('./TestSet.csv')

def one_hot_encode(y,n_classes):
    #y - shape:(N,) -  0,1
    #return mat: Nx2
    one_hot_mat=np.zeros((y.shape[0],n_classes))
    for label_idx in range(y.shape[0]):
        label=y[label_idx]
        one_hot_mat[label_idx,label]=1    
    return one_hot_mat

def get_next_train(i,batch_size):
    global X_train,y_train
    #return [batch_size,time_steps] [batch_size,3]
    if(i+batch_size>n_samples):
       y=one_hot_encode(y_train[i:n_samples],n_classes)
       x=X_train[i:n_samples,:]
       #x=x.reshape(*x.shape,1)
       return (x,y)
    else:   
       y=one_hot_encode(y_train[i:i+batch_size],n_classes) 
       x=X_train[i:i+batch_size,:]
       #x=x.reshape(*x.shape,1)
       return (x,y)

def get_next_test(i,batch_size):
    global X_test,y_test
    if(i+batch_size>n_test):
       y=one_hot_encode(y_test[i:n_test],n_classes)
       x=X_test[i:n_test,:]
       #x=x.reshape(*x.shape,1)
       return (x,y)
    else:   
       y=one_hot_encode(y_test[i:i+batch_size],n_classes)
       x=X_test[i:n_test,:]
       #x=x.reshape(*x.shape,1)
       return (x,y)

def biRNN():
    inputs=tf.reshape(X,[-1],name='flattened_input')
    embedded_inputs = tf.nn.embedding_lookup(word_embeddings,inputs)

    embedded_inputs=tf.reshape(embedded_inputs,[-1,time_steps,128])
    embedded_inputs=tf.unstack(embedded_inputs,time_steps,1)

    lstm_layer_fw=rnn.BasicLSTMCell(num_units,forget_bias=1)
    lstm_layer_bw=rnn.BasicLSTMCell(num_units,forget_bias=1)
    conc_outputs,final_state_fw,final_state_bw=rnn.stack_bidirectional_rnn(lstm_layer_fw,lstm_layer_bw,embedded_inputs,\
                dtype='float32')

    return tf.concat(final_state_fw,final_state_bw)
    
#learn word embeddings for length of vocab
word_embeddings = tf.get_variable("word_embeddings",[vocabulary_size, embedding_size])
X=tf.placeholder("int32",[None,time_steps])
Y=tf.placeholder("float32",[None,n_classes])

W=tf.Variable(tf.random_normal([num_units,n_classes]),name='weight') #(128x10)
b=tf.Variable(tf.random_normal([n_classes]),name='bias') #(1x10)
#this is converted to a list of values  of shape [1,batch_size x time_steps x 1]
inputs=tf.reshape(X,[-1],name='flattened_input')
embedded_inputs = tf.nn.embedding_lookup(word_embeddings,inputs)

#reshape to list of timesteps of tensors [batch_size,128]
embedded_inputs=tf.reshape(embedded_inputs,[-1,time_steps,128])
embedded_inputs=tf.unstack(embedded_inputs,time_steps,1)
#output shape list of timesteps tensors of shape [batch_size,128]
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,embedded_inputs,dtype='float32')
#outputs of shape (batch_size,128)

pred=tf.add(tf.matmul(outputs[-1],W,transpose_a=False,transpose_b=False),b)
#shape : (batch_size,10)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=Y))
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

target_pred=tf.argmax(tf.nn.softmax(pred),1)
accuracy=tf.reduce_mean(tf.cast(tf.equal(target_pred,tf.argmax(Y,1)),tf.float32))
#init variables
init=tf.global_variables_initializer()

train_epoch_losses=[]
train_epoch_acc=[]
train_batch_losses=[]
train_batch_acc=[]

with tf.Session() as sess:
     sess.run(init)
     saver = tf.train.Saver()

     if(os.path.exists('./model_rnn.ckpt.meta')):
         saver = tf.train.import_meta_graph('./model_rnn.ckpt.meta')
         saver.restore(sess,tf.train.latest_checkpoint('./'))
         print("Model restored.")

     for epoch in range(training_epochs):
         i=0
         n_steps=n_samples//batch_size #1 to n_batches ~ 0 to n_batches-1 
         total_steps=n_steps
         loss_=0
         acc=0 
         while(n_steps):    
            x,y =get_next_train(i,batch_size)
            l,_,acc=sess.run([cost,optimizer,accuracy],feed_dict={X:x,Y:y})
            loss_=loss_+l  #average loss per batch.
   
            train_batch_losses.append(l) 
            train_batch_acc.append(acc) 
            #print("Train batch loss:",loss_)
            #print("Train batch acc:",acc)
            i=i+batch_size
            n_steps=n_steps-1

         print("Train epoch loss:",epoch,loss_/total_steps) #overall average loss across all steps.
         train_epoch_losses.append(loss_/total_steps)

         if(epoch%2==0): 
             save_path = saver.save(sess, "./model_rnn.ckpt")
             print("Model saved in file: %s" % save_path) 
         
             x,y=get_next_train(0,n_samples) 
             acc,preds=sess.run([accuracy,target_pred],feed_dict={X:x,Y:y})

             train_epoch_acc.append(acc) 
             print(sum(np.argmax(y,axis=1).tolist()),sum(preds))
             print("Train acc:",epoch,acc)

             i=0
             #give them all at once.
             batch_size_=n_test
             n_steps=n_test//batch_size_
             total_steps=n_steps
             loss_=0
             while(n_steps):
                 x,y =get_next_test(i,batch_size_)
                 preds,l,acc=sess.run([target_pred,cost,accuracy],feed_dict={X:x,Y:y})
                 loss_=loss_+l #average loss per batch.
                 print("Testset loss:",epoch,loss_)
                 print("Testset accuracy:",epoch,acc)
                 i=i+batch_size
                 n_steps=n_steps-1
                 pred_df=pd.DataFrame(x.reshape(x.shape[0],-1))
                 pred_df['label']=np.argmax(y,axis=1).tolist()
                 pred_df['preds']=preds.tolist()
                 print(sum(pred_df['label']),sum( pred_df['preds']))
                 pred_df.to_csv('./Results/Glutenberg_Corpus_'+str(90+epoch)+'.csv') 
             print("Testing loss: ",loss_/total_steps) #overall average loss across all steps.
