from __future__ import division
import pandas as pd
import numpy as np
from collections import defaultdict,OrderedDict
from sklearn.model_selection import train_test_split
import pickle
from keras.preprocessing import sequence
import tensorflow as tf
from tensorflow.contrib import rnn
import os,time
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler as minmax

tf.reset_default_graph()

learning_rate = 0.0001
training_epochs = 20#10000
display_step = 5


def find_where_nans(df):
    for i in range(df.shape[0]):
        if(df.iloc[i,:].hasnans):
          print(i,df.iloc[i,:])

start=time.time()
df=pd.read_csv('./YTF_dataset_nextf.csv')
print("Time to read: ",time.time()-start)

print("Removing Labels..")
d_=df[df['Labels']=='Labels']
df=df.drop(d_.index).reset_index()
print(df.shape)

df=df.drop(labels=['Unnamed: 0','Labels','index'],axis=1)
print(df.shape)
'''
print("Removing nans..")
find_where_nans(df)
df=df.dropna(axis=0,how='any').reset_index()
'''
df['idx']=list(range(df.shape[0]))
print(df.shape)

#indices run from 0...N
def is_5th_frame(row):
    return ((row['idx']+1)%5==0)

def is_not5th_frame(row):
    return ((row['idx']+1)%5!=0)

time_steps=4 
num_units=128
batch_size=25

#build x_train and y_train
y_train=df[df.apply(is_5th_frame,axis=1)]
y_train=y_train.drop(labels=['idx'],axis=1)
y_train=y_train.as_matrix()
y_train=minmax().fit_transform(y_train)

#reshape this to [N,64*64*3]
y_train=y_train.reshape(-1,64*64*3)

X_train=df[df.apply(is_not5th_frame,axis=1)]
X_train=X_train.drop(labels=['idx'],axis=1)
X_train=X_train.as_matrix()
X_train=minmax().fit_transform(X_train)

#reshape this into [N,time_steps,64*64*3]
X_train=X_train.reshape(-1,time_steps,64*64*3)

#both will be of same shape - 2874
print(X_train.shape,y_train.shape)
n_samples=X_train.shape[0]

#get test data
start=time.time()
df=pd.read_csv('./YTF_dataset_nextf_test.csv')
print("Time to read: ",time.time()-start)

print("Removing Labels..")
d_=df[df['Labels']=='Labels']
df=df.drop(d_.index).reset_index()
print(df.shape)

df=df.drop(labels=['Unnamed: 0','Labels','index'],axis=1)
print(df.shape)
df['idx']=list(range(df.shape[0]))
print(df.shape)

y_test=df[df.apply(is_5th_frame,axis=1)]
y_test=y_test.drop(labels=['idx'],axis=1)
y_test=y_test.as_matrix()
y_test=minmax().fit_transform(y_test)
#reshape this to [N,64*64*3]
y_test=y_test.reshape(-1,64*64*3)

X_test=df[df.apply(is_not5th_frame,axis=1)]
X_test=X_test.drop(labels=['idx'],axis=1)
X_test=X_test.as_matrix()
X_test=minmax().fit_transform(X_test)
#reshape this into [N,time_steps,64*64*3]
X_test=X_test.reshape(-1,time_steps,64*64*3)

#both will be of same shape - 2874
print(X_test.shape,y_test.shape)
n_test=X_test.shape[0]

def get_next_train(i,batch_size):
    global X_train,y_train
    #return [batch_size,time_steps,64*64*3] [batch_size,64*64*3]
    if(i+batch_size>n_samples):
       y=y_train[i:n_samples,:]
       x=X_train[i:n_samples,:,:]
       return (x,y)
    else:   
       y=y_train[i:i+batch_size,:]
       x=X_train[i:i+batch_size,:,:]
       return (x,y)

def get_next_test(i,batch_size):
    global X_test,y_test
    #return [batch_size,time_steps,64*64*3] [batch_size,64*64*3]
    if(i+batch_size>n_samples):
       y=y_test[i:n_samples,:]
       x=X_test[i:n_samples,:,:]
       return (x,y)
    else:   
       y=y_test[i:i+batch_size,:]
       x=X_test[i:i+batch_size,:,:]
       return (x,y)

X=tf.placeholder("float32",[None,time_steps,64*64*3])
Y=tf.placeholder("float32",[None,64*64*3])

W=tf.Variable(tf.random_normal([num_units,64*64*3]),name='weight') #(128x10)
b=tf.Variable(tf.random_normal([64*64*3]),name='bias') #(1x10)

inputs=tf.unstack(X,time_steps,1)
#output shape list of timesteps tensors of shape [batch_size,128]
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,inputs,dtype='float32')
#outputs of shape (batch_size,128)

pred=tf.add(tf.matmul(outputs[-1],W,transpose_a=False,transpose_b=False),b)
#shape : (batch_size,64*64*3)
cost=tf.reduce_mean(tf.abs(pred - Y))
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

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
         while(n_steps+1):    
            x,y =get_next_train(i,batch_size)
            #print(x.shape,y.shape)
            l,_=sess.run([cost,optimizer],feed_dict={X:x,Y:y})
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
             #print(x.shape,y.shape)
             loss_=sess.run(cost,feed_dict={X:x,Y:y})
             print("Overall train loss: ",loss_)

             x,y=get_next_test(0,n_test) 
             #print(x.shape,y.shape)
             loss_,preds=sess.run([cost,pred],feed_dict={X:x,Y:y})
             print("Overall test loss: ",loss_)
             test_df=pd.DataFrame(preds)
             test_df.to_csv('./testPredictions.csv') 
