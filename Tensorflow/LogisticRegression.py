from __future__ import print_function,division
import tensorflow as tf
import numpy as np
import pandas as pd
import os.path 

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50
batch_size=50

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
n_classes=10 #number of distinct classes.
n_samples=X_train.shape[0]
n_test=X_test.shape[0]
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
    if(i+batch_size>n_samples):
       y=one_hot_encode(y_train[i:n_samples],n_classes)
       return (X_train[i:n_samples,:].reshape(-1,n_cols),y)
    else:   
       y=one_hot_encode(y_train[i:i+batch_size],n_classes) 
       return (X_train[i:i+batch_size,:].reshape(-1,n_cols),y)

def get_next_test(i,batch_size):
    if(i+batch_size>n_test):
       y=one_hot_encode(y_test[i:n_test],n_classes)
       return (X_test[i:n_test,:].reshape(-1,n_cols),y)
    else:   
       y=one_hot_encode(y_test[i:i+batch_size],n_classes)
       return (X_test[i:i+batch_size,:].reshape(-1,n_cols),y)

#weigts and biases as variables.
X=tf.placeholder("float64")
Y=tf.placeholder("float64")

W=tf.Variable(np.random.randn(n_cols,n_classes),name='weight')
b=tf.Variable(np.random.randn(1,n_classes),name='bias')

#define functions like pred,loss.
#pred - W.T*x+b or softmax(X.W+b) ->gives a prob dist over n_classes
#cost - np.mean((pred-y)**2)
pred=tf.nn.softmax(tf.add(tf.matmul(X,W,transpose_a=False,transpose_b=False),b))
cost=tf.reduce_mean(tf.reduce_sum(tf.multiply(Y,pred),axis=1))
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

target_pred=tf.argmax(pred,1)
accuracy=tf.reduce_mean(tf.cast(tf.equal(target_pred,tf.argmax(Y,1)),tf.float32))
#init variables
init=tf.global_variables_initializer()

with tf.Session() as sess:
     sess.run(init)
     saver = tf.train.Saver()

     if(os.path.exists('./model_logreg.ckpt.meta')):
         saver = tf.train.import_meta_graph('./model_logreg.ckpt.meta')
         saver.restore(sess,tf.train.latest_checkpoint('./'))
         print("Model restored.")
         print(W.eval().shape)

     for epoch in range(training_epochs):
         i=0
         n_steps=n_samples//batch_size #1 to n_batches ~ 0 to n_batches-1 
         total_steps=n_steps
         loss_=0
         while(n_steps):    
            x,y =get_next_train(i,batch_size)
            l,_=sess.run([cost,optimizer],feed_dict={X:x,Y:y})
            loss_=loss_+l  #average loss per batch.
            #print("Train batch loss:",i,loss_)
            i=i+batch_size
            n_steps=n_steps-1

         print("Train epoch loss:",epoch,loss_/total_steps) #overall average loss across all steps.
 
         if(epoch%100==0): 
             save_path = saver.save(sess, "./model_logreg.ckpt")
             print("Model saved in file: %s" % save_path) 
             i=0
             #give them all at once.
             batch_size_=n_test
             n_steps=n_test//batch_size_
             total_steps=n_steps
             loss_=0
             while(n_steps):
                 x,y =get_next_test(i,batch_size_)
                 preds,l=sess.run([target_pred,cost],feed_dict={X:x,Y:y})
                 loss_=loss_+l #average loss per batch.
                 #print("Test batch loss:",i,loss_)
                 i=i+batch_size
                 n_steps=n_steps-1
                 pred_df=pd.DataFrame(x)
                 pred_df['label']=np.argmax(y,axis=1).tolist()
                 pred_df['preds']=preds.tolist()
                 pred_df.to_csv('./Results/Fashion_Mnist_'+str(epoch)+'.csv') 
             print("Testing loss: ",loss_/total_steps) #overall average loss across all steps.

     
