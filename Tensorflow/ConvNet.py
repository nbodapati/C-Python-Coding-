from __future__ import absolute_import,division
from __future__ import print_function

from sklearn.preprocessing import MinMaxScaler as minmax
import numpy as np
import pandas as pd
import os.path 
import pickle

import os
import sys
import tensorflow as tf

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 300.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

training_epochs = 1001#10000
display_step = 50
n_classes=10
def load_data(train_file='./fashion-mnist_train.csv',test_file='fashion-mnist_test.csv'):
    #dataset - labels #10 - column1 of the csv file.
    train_df=pd.read_csv(train_file)
    test_df=pd.read_csv(test_file)
    train_labels=train_df.iloc[:,0].as_matrix()
    train_features=train_df.iloc[:,1:].as_matrix()
    train_features=minmax().fit_transform(train_features)
    test_labels=test_df.iloc[:,0].as_matrix()
    test_features=test_df.iloc[:,1:].as_matrix()
    test_features=minmax().fit_transform(test_features)
    return (train_features,train_labels,test_features,test_labels)

X_train,y_train,X_test,y_test= load_data()
n_cols=X_train.shape[1] #784
n_samples=X_train.shape[0]
n_test=X_test.shape[0]

#X_train should be of shape (batch_size,image_rows,image_cols) or 
#(batch_size,time_steps,n_input)
X_train=X_train.reshape(n_samples,28,28,1) #(50000x28x28x1)
X_test=X_test.reshape(n_test,28,28,1)  #(10000x28x28x1)

X=tf.placeholder("float32",[None,28,28,1])
Y=tf.placeholder("float32",[None,10])
batch_size=tf.placeholder(tf.int32,[])

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
       return (X_train[i:n_samples,:,:,:],y)
    else:   
       y=one_hot_encode(y_train[i:i+batch_size],n_classes) 
       return (X_train[i:i+batch_size,:,:,:],y)

def get_next_test(i,batch_size):
    global X_test,y_test
    if(i+batch_size>n_test):
       y=one_hot_encode(y_test[i:n_test],n_classes)
       return (X_test[i:n_test,:,:,:],y)
    else:   
       y=one_hot_encode(y_test[i:i+batch_size],n_classes)
       return (X_test[i:i+batch_size,:,:,:],y)


def create_variable(name, shape,init_):
  """
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape,\
                   initializer=init_,\
                   dtype=tf.float32)
  return var

def create_variable_with_decay(name,shape,std,wd):
    init_=tf.truncated_normal_initializer(stddev=std, dtype=tf.float32)
    var=create_variable(name, shape,init_)
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    return var

#inputs are 3-D tensors of shapes [batch_size,image_size,image_size,num_channels]
#[batch_size,28,28,1] "NHWC"
with tf.variable_scope('conv1') as scope:
    kernel = create_variable_with_decay('weights',
                                         shape=[3, 3, 1, 64], #3x3x1 sized 64 filters
                                         std=5e-2,
                                         wd=0.01)

    conv = tf.nn.conv2d(X, kernel, [1, 1, 1, 1], padding='SAME') #[1,1,1,1] - stride length in each dim.
    #output shape: [batch_size,28,28,64]
    biases = create_variable('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases) 
    conv1 = tf.nn.relu(pre_activation, name=scope.name) #output shape [batch_size,28,28,64]; name:"conv1"
#pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1') #ksize - kernal size for each dimension.
#norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
    reshape1 = tf.reshape(norm1, [batch_size, -1])  #flatten it to [batch_size,*(other dims)]
    dim = reshape1.get_shape()[1].value

with tf.variable_scope('fc1') as scope:
    weights = create_variable_with_decay('weights', shape=[14*14*64, 192],
                                          std=0.04, wd=0.004)
    biases = create_variable('biases', [192], tf.constant_initializer(0.1))
    fc1 = tf.nn.relu(tf.matmul(reshape1, weights) + biases, name=scope.name)  #output [batch_size,192]

with tf.variable_scope('softmax_linear') as scope:
    weights = create_variable_with_decay('weights', [192,10],
                                          std=1/192.0, wd=0.0)
    biases = create_variable('biases', [10],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(fc1, weights), biases, name=scope.name)
    #output shape: [batch_size,10]

#softmax_cross entropy loss
#labels - tensor shape [batch_size,n_classes]
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=Y, logits=softmax_linear, name='cross_entropy_per_example')
loss=cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

b_size=50
num_batches_per_epoch = n_samples//b_size
decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY) #*300

global_step=tf.Variable(initial_value=0,dtype=tf.int32,trainable=False)
lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)

optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
#grads = opt.compute_gradients(loss)
# Apply gradients.
#apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)  

# Track the moving averages of all trainable variables.
variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())

target_pred=tf.argmax(tf.nn.softmax(softmax_linear),1)
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

     if(os.path.exists('./model_cnn.ckpt.meta')):
         saver = tf.train.import_meta_graph('./model_cnn.ckpt.meta')
         saver.restore(sess,tf.train.latest_checkpoint('./'))
         print("Model restored.")

     b_size=50
     for epoch in range(training_epochs):
         i=0
         n_steps=n_samples//b_size #1 to n_batches ~ 0 to n_batches-1 
         total_steps=n_steps
         loss_=0
         acc=0 
         while(n_steps):    
            x,y =get_next_train(i,b_size)
            l,_,acc,_=sess.run([loss,optimizer,accuracy,variables_averages_op],feed_dict={X:x,Y:y,batch_size:b_size})
            loss_=loss_+l  #average loss per batch.
   
            train_batch_losses.append(l) 
            train_batch_acc.append(acc) 
            #print("Train batch loss:",loss_)
            #print("Train batch acc:",acc)
            i=i+b_size
            n_steps=n_steps-1
         print("Train epoch loss:",epoch,loss_/total_steps) #overall average loss across all steps.
         train_epoch_losses.append(loss_/total_steps)

         if(epoch%10==0): 
             save_path = saver.save(sess, "./model_cnn.ckpt")
             print("Model saved in file: %s" % save_path) 
             
             x,y=get_next_train(0,n_samples) 
             acc=sess.run(accuracy,feed_dict={X:x,Y:y,batch_size:n_samples})

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
                 preds,l,acc=sess.run([target_pred,loss,accuracy],feed_dict={X:x,Y:y,batch_size:n_test})
                 loss_=loss_+l #average loss per batch.
                 print("Testset loss:",epoch,loss_)
                 print("Testset accuracy:",epoch,acc)
                 i=i+batch_size_
                 n_steps=n_steps-1
                 pred_df=pd.DataFrame(x.reshape(x.shape[0],-1))
                 pred_df['label']=np.argmax(y,axis=1).tolist()
                 pred_df['preds']=preds.tolist()
                 pred_df.to_csv('./Results_CNN/Fashion_Mnist_'+str(epoch)+'.csv') 
     
     train_dict=dict(epoch_loses=train_epoch_losses,epoch_acc=train_epoch_acc,batch_loses=train_batch_losses,\
                         batch_acc=train_batch_acc)           
     pickle.dump(train_dict,open('./Results_CNN/train_stats.pkl','wb'))



