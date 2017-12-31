from __future__ import print_function,division
import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

X_train=np.random.randn(15,5)
y_train=np.random.randn(15,1)
n_samples=X_train.shape[0]

X_test=np.random.randn(15,5)
y_test=np.random.randn(15,1)
n_test=X_test.shape[0]

def get_next_data(i):
    return (X_train[i,:].reshape(1,-1),y_train[i])

def get_next_test(i):
    return (X_test[i,:],y_test[i])

#weigts and biases as variables.
X=tf.placeholder("float64")
Y=tf.placeholder("float64")

W=tf.Variable(np.random.randn(5,1),name='weight')
b=tf.Variable(np.random.randn(1,1),name='bias')

#define functions like pred,loss.
#pred - W.T*x+b or X.W+b
#cost - np.mean((pred-y)**2)
pred=tf.add(tf.matmul(X,W,transpose_a=False,transpose_b=False),b)
cost=tf.reduce_sum(tf.pow(pred-Y,2))/n_samples
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#init variables
init=tf.global_variables_initializer()

with tf.Session() as sess:
     sess.run(init)
     for epoch in range(training_epochs):
         i=0
         loss_=0
         while(1):    
            x,y =get_next_data(i)
            l,_=sess.run([cost,optimizer],feed_dict={X:x,Y:y})
            loss_=loss_+l
            i=i+1
            if(i==n_samples):
               break
         print("Epoch loss:",epoch,loss_/n_samples)
 
         if(epoch%10==0): 
            testing_loss=sess.run(\
            tf.reduce_sum(tf.pow(pred-Y,2))/n_test,\
                    feed_dict={X:X_test,Y:y_test})  
            print("TEsting loss: ",testing_loss)
            print('loss diff',abs(loss_/n_samples-testing_loss))


