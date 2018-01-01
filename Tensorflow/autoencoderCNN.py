from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import pandas as pd
import os.path
from sklearn.preprocessing import MinMaxScaler as minmax

starter_learning_rate =0.001
learning_rate=starter_learning_rate
global_step = tf.Variable(0, trainable=False)
learning_rate_ = tf.train.exponential_decay(starter_learning_rate, global_step,\
                 1000, 0.96, staircase=True)

num_epochs = 15000
display_step = 10

num_hidden_1 =50 # 1st layer num features
num_hidden_2 =2 
num_input = 192 # 4 is the input feature size

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

#Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.abs(y_true - y_pred)) #MAE
opt=tf.train.AdamOptimizer(learning_rate_)
optimizer = opt.minimize(loss)#,\
             #global_step=global_step) #this will increment global_step on 
                                      #each optim update.

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

#load data from text file and 
#split to train,val sets.
def get_data(file_='./Fashion_Mnist_Vectors.csv'):
    df=pd.read_csv(file_)
    df2=pd.read_csv('./Fashion_Mnist_10.csv')
    labels=df2['label'].as_matrix()
    dataset=df.iloc[:,1:].as_matrix()
    
    return (dataset,labels)

X_train,y_train=get_data()
X_train=minmax().fit_transform(X_train)
print("Shape of X_train and y_train:",X_train.shape,y_train.shape)
batch_size=X_train.shape[0]

num_batches=X_train.shape[0]//batch_size
num_steps=num_batches*num_epochs
print(num_steps)

def train_next_batch(i,batch_size):
    global X_train
    if(i+batch_size > X_train.shape[0]):
       return X_train[i:X_train.shape[0],:]
    else:
       return X_train[i:i+batch_size,:]

# Start Training
# Start a new TF session
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    #to save the variables - weights and biases
    saver = tf.train.Saver()
    # Training
    # Restore variables from disk.
    if(os.path.exists('./model_autoenc.ckpt.meta')):
        #saver.restore(sess, "./model_autoenc.ckpt")
        saver = tf.train.import_meta_graph('./model_autoenc.ckpt.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        print("Model restored.")

    start_i=0
    for i in range(0, num_steps+1):
        #Prepare Data
        #Get the next batch of MNIST data (only images are needed, not labels)
        batch_x = train_next_batch(start_i,batch_size)
        #Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X:batch_x})

        #Display logs per step
        if i % display_step == 0 or i == 1:
           print('Step %i:New Minibatch Loss: %f' % (i, l))
           start_i=(start_i+batch_size)%X_train.shape[0] #this can go from 0 - train_size 
        if i % 10000 == 0:
           save_path = saver.save(sess, "./model_autoenc.ckpt")
           print("Model saved in file: %s" % save_path) 

    #Testing
    batch_size=X_train.shape[0]
    steps=X_train.shape[0]//batch_size
    reduced_X_train=np.zeros((X_train.shape[0],num_hidden_2))

    start_i=0
    for i in range(0,steps):
        batch_x= train_next_batch(start_i,batch_size)
        #Encode and decode the digit image
        g,l = sess.run([encoder_op,loss], feed_dict={X: batch_x})
        reduced_X_train[start_i:start_i+batch_size,:]=g
        start_i=(start_i+batch_size)%X_train.shape[0] #this can go from 0 - train_size 
        print(i,g.shape,l)

    print(reduced_X_train.shape)
    df_reduced=pd.DataFrame(reduced_X_train)
    df_reduced.to_csv('./Fashion_MNIST_reduced.csv')
        


