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
from sklearn.utils import shuffle
import time
import warnings
warnings.filterwarnings('ignore')

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 300.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.00001       # Initial learning rate.

training_epochs = 10
display_step = 50

max_train_per_class=50
num_classes=None
name2Id_map=None

def get_distribution(labels):
    labels=pd.Series(labels)
    #print(labels.value_counts())
    return labels.value_counts().to_dict()

def print_dict(d):
    for k,v in d.items():
        print(k,v) 

def label_to_int(labels):
    return [name2Id_map[label] for label in labels] 

def one_hot_encode(y,n_classes):
    #y - shape:(N,) -  0 to n_classes
    #return mat: N x n_classes
    one_hot_mat=np.zeros((y.shape[0],n_classes))
    for label_idx in range(y.shape[0]):
        label=y[label_idx]
        one_hot_mat[label_idx,label]=1    
    return one_hot_mat

def get_next_train(i,batch_size):
    global X_train,y_train,num_classes
    #shapes: [N,h,w,c] [N]
    if(i+batch_size>n_samples):
       y=one_hot_encode(y_train[i:n_samples],num_classes)
       return (X_train[i:n_samples,:,:,:],y)
    else:   
       y=one_hot_encode(y_train[i:i+batch_size],num_classes) 
       return (X_train[i:i+batch_size,:,:,:],y)

def get_next_test(i,batch_size):
    global X_test,y_test
    if(i+batch_size>n_test):
       y=one_hot_encode(y_test[i:n_test],num_classes)
       return (X_test[i:n_test,:,:,:],y)
    else:   
       y=one_hot_encode(y_test[i:i+batch_size],num_classes)
       return (X_test[i:i+batch_size,:,:,:],y)


def load_data(csv_file='./YTF_dataset.csv'):
    global num_classes,name2Id_map
    #dataset - labels #10 - column1 of the csv file.
    print("Loading data..")
    start=time.time()
    data=pd.read_csv(csv_file)
    print("Time to read..",time.time()-start)
   
    #remove junk data
    i=data.loc[data['Labels']=='Labels',:].index
    data=data.drop(i,axis=0).reset_index()
    
    #has keys 'Images' of shape (batch_size,h,w,c)
    #and 'Labels' of shape (batch_size,) 
    dist_=get_distribution(data['Labels'])
    #to restrict num_train per category.         
    num_classes=len(dist_.keys())
    name2Id_map=dict(list(zip(data['Labels'].unique(),\
                          list(range(num_classes)))))
    #print_dict(dist_) 
    #print_dict(name2Id_map)
    return data.iloc[:,:-2].as_matrix(),data['Labels']

def train_test_split(X,y):
    global max_train_per_class
    max_cat=max_train_per_class
    df=pd.DataFrame(X)
    df['labels']=y

    grouped=df.groupby(['labels'])
    def get_grp(label):
        return grouped.get_group((label))

    unique_labels=np.unique(y)
    train_df=get_grp(unique_labels[0])

    train_df=shuffle(train_df)
    test_df=train_df.iloc[max_cat+1:,:]
    test_df.loc[:,'label']=unique_labels[0]
    train_df=train_df.iloc[:max_cat+1,:]
    train_df.loc[:,'label']=unique_labels[0]

    for l in unique_labels[1:]:
        df=get_grp(l)
        df=shuffle(df)
        test_df_=df.iloc[max_cat+1:,:]
        test_df_['label']=l
        train_df_=df.iloc[:max_cat+1,:]
        train_df_['label']=l
        train_df=pd.concat([train_df,train_df_]) 
        test_df=pd.concat([test_df,test_df_]) 

    return train_df.iloc[:,1:-1].as_matrix(),train_df['label'].as_matrix(),test_df.iloc[:,1:-1].as_matrix(),\
                        test_df['label'].as_matrix()
 
#load_data before anything else.
x,y=load_data()
#x=np.array([np.array(xi) for xi in x])
y=label_to_int(y)
#print(y[:100])
#turn labels to int.
X_train,y_train,X_test,y_test=train_test_split(x,y)
X_train=X_train.reshape(-1,64,64,3)
X_test=X_test.reshape(-1,64,64,3)

y_train=np.asarray(y_train,dtype=int)
y_test=np.asarray(y_test,dtype=int)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
n_samples=X_train.shape[0]
n_test=X_test.shape[0]

print(num_classes)     
X=tf.placeholder("float32",[None,64,64,3])
Y=tf.placeholder("float32",[None,num_classes])
batch_size=tf.placeholder(tf.int32,[])
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
#[batch_size,128,128,3] "NHWC"
with tf.variable_scope('conv1') as scope:
    kernel = create_variable_with_decay('weights',
                                         shape=[5, 5, 3, 64], #3x3x1 sized 64 filters
                                         std=5e-2,
                                         wd=0.01)

    conv = tf.nn.conv2d(X, kernel, [1, 1, 1, 1], padding='SAME') #[1,1,1,1] - stride length in each dim.
    #output shape: [batch_size,128,128,64]
    biases = create_variable('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases) 
    conv1 = tf.nn.relu(pre_activation, name=scope.name) #output shape [batch_size,28,28,64]; name:"conv1"
#pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1') #ksize - kernal size for each dimension.
#norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

    #output shape [b_s,32,32,64] 
with tf.variable_scope('conv2') as scope:
    kernel = create_variable_with_decay('weights2',
                                         shape=[5, 5,64, 64], #3x3x1 sized 64 filters
                                         std=5e-2,
                                         wd=0.01)

    conv2 = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME') #[1,1,1,1] - stride length in each dim.
    #output shape: [batch_size,16,16,64]
    biases2 = create_variable('biases2', [64], tf.constant_initializer(0.0))
    pre_activation2 = tf.nn.bias_add(conv2, biases2) 
    conv2 = tf.nn.relu(pre_activation2, name=scope.name) #output shape [batch_size,28,28,64]; name:"conv1"
#pool1
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2') #ksize - kernal size for each dimension.
#norm1
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
    #output shape [b_s,8,8,64]
    reshape2 = tf.reshape(norm2, [batch_size, -1])  #flatten it to [batch_size,*(other dims)]
    dim2 = reshape2.get_shape()[1].value
    #output shape [b_s,16,16,64]
''' 
with tf.variable_scope('conv3') as scope:
    kernel = create_variable_with_decay('weights',
                                         shape=[5, 5, 3, 64], #3x3x1 sized 64 filters
                                         std=5e-2,
                                         wd=0.01)

    conv3 = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='SAME') #[1,1,1,1] - stride length in each dim.
    #output shape: [batch_size,4,4,64]
    biases3 = create_variable('biases', [64], tf.constant_initializer(0.0))
    pre_activation3 = tf.nn.bias_add(conv3, biases3) 
    conv3 = tf.nn.relu(pre_activation3, name=scope.name) #output shape [batch_size,28,28,64]; name:"conv1"
#pool1
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2') #ksize - kernal size for each dimension.
#norm1
    norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
'''

with tf.variable_scope('softmax_linear') as scope:
    weights = create_variable_with_decay('weights_', [16*16*64,num_classes],
                                          std=1/192.0, wd=0.0)
    biases = create_variable('biases_', [num_classes],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(reshape2, weights), biases, name=scope.name)
    #output shape: [batch_size,num_classes]

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

optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
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
            l,_,acc,ce,pred=sess.run([loss,optimizer,accuracy,cross_entropy,softmax_linear],\
                                     feed_dict={X:x,Y:y,batch_size:b_size})
            loss_=loss_+l  #average loss per batch.
   
            train_batch_losses.append(l) 
            train_batch_acc.append(acc) 
            #print("Train batch loss:",loss_)
            #print("Train batch acc:",acc)
            #print("x:",x)
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
                 preds,l,acc,fc1_outputs=sess.run([target_pred,loss,accuracy,softmax_linear],feed_dict={X:x,Y:y,batch_size:n_test})
                 loss_=loss_+l #average loss per batch.
                 print("Testset loss:",epoch,loss_)
                 print("Testset accuracy:",epoch,acc)
                 i=i+batch_size_
                 n_steps=n_steps-1
                 pred_df=pd.DataFrame(x.reshape(x.shape[0],-1))
                 pred_df['label']=np.argmax(y,axis=1).tolist()
                 pred_df['preds']=preds.tolist()
                 pred_df.to_csv('./Results_CNN/YTF_faces_'+str(epoch)+'.csv') 
                 
                 #test_rep_df=pd.DataFrame(fc1_outputs)
                 #test_rep_df.to_csv('./Results_CNN/Fashion_Mnist_Vectors.csv') 
                    
     train_dict=dict(epoch_loses=train_epoch_losses,epoch_acc=train_epoch_acc,batch_loses=train_batch_losses,\
                         batch_acc=train_batch_acc)           
     pickle.dump(train_dict,open('./Results_CNN/train_stats.pkl','wb'))



