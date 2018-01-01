from __future__ import print_function,division
import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np
import pandas as pd
import os.path 
import pickle

# Parameters
learning_rate = 0.01
training_epochs = 11#10000
display_step = 50
batch_size=50

time_steps=28
num_units=128
n_classes=10
n_input=28

def load_data(train_file='./fashion-mnist_train.csv',test_file='fashion-mnist_test.csv'):
    #dataset - labels #10 - column1 of the csv file.
    train_df=pd.read_csv(train_file)
    test_df=pd.read_csv(test_file)
    train_labels=train_df.iloc[:,0].as_matrix()
    train_features=train_df.iloc[:,1:].as_matrix()
    test_labels=test_df.iloc[:,0].as_matrix()
    test_features=test_df.iloc[:,1:].as_matrix()
    return (train_features,train_labels,test_features,test_labels)

X_train,y_train,X_test,y_test= load_data()
n_cols=X_train.shape[1] #784
n_samples=X_train.shape[0]
n_test=X_test.shape[0]

#X_train should be of shape (batch_size,image_rows,image_cols) or 
#(batch_size,time_steps,n_input)
X_train=X_train.reshape(n_samples,n_input,n_input) #(50000x28x28)
X_test=X_test.reshape(n_test,n_input,n_input)  #(10000x28x28)


print("Train shapes: ",X_train.shape,y_train.shape)
print("Test shapes: ",X_test.shape,y_test.shape)

def one_hot_encode(y,n_classes):
    #y - shape:(N,) -  0 to 9
    #return mat: Nx10
    one_hot_mat=np.zeros((y.shape[0],n_classes))
    for label_idx in range(y.shape[0]):
        label=y[label_idx]
        one_hot_mat[label_idx,label]=1    
    return one_hot_mat

def get_next_train(i,batch_size):
    global X_train,y_train
    if(i+batch_size>n_samples):
       y=one_hot_encode(y_train[i:n_samples],n_classes)
       return (X_train[i:n_samples,:,:],y)
    else:   
       y=one_hot_encode(y_train[i:i+batch_size],n_classes) 
       return (X_train[i:i+batch_size,:,:],y)

def get_next_test(i,batch_size):
    global X_test,y_test
    if(i+batch_size>n_test):
       y=one_hot_encode(y_test[i:n_test],n_classes)
       return (X_test[i:n_test,:,:],y)
    else:   
       y=one_hot_encode(y_test[i:i+batch_size],n_classes)
       return (X_test[i:i+batch_size,:,:],y)

#weigts and biases as variables.
X=tf.placeholder("float32",[None,time_steps,n_input])
Y=tf.placeholder("float32",[None,n_classes])

W=tf.Variable(tf.random_normal([num_units,n_classes]),name='weight') #(128x10)
b=tf.Variable(tf.random_normal([n_classes]),name='bias') #(1x10)

#this is converted to a list of "timesteps" tensors of shape (batch_size,n_input)
inputs=tf.unstack(X,time_steps,1)
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,inputs,dtype='float32')
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
         print(W.eval().shape)

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

         if(epoch%10==0): 
             save_path = saver.save(sess, "./model_rnn.ckpt")
             print("Model saved in file: %s" % save_path) 
         
             x,y=get_next_train(0,n_samples) 
             acc=sess.run(accuracy,feed_dict={X:x,Y:y})

             train_epoch_acc.append(acc) 
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
                 pred_df.to_csv('./Results_RNN/Fashion_Mnist_'+str(90+epoch)+'.csv') 
             print("Testing loss: ",loss_/total_steps) #overall average loss across all steps.


     train_dict=dict(epoch_loses=train_epoch_losses,epoch_acc=train_epoch_acc,batch_loses=train_batch_losses,\
                         batch_acc=train_batch_acc)           
     pickle.dump(train_dict,open('train_stats.pkl','wb'))
