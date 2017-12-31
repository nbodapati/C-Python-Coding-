#Practise basic examples from 
#https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/1_Introduction/basic_operations.ipynb

import tensorflow as tf

a=tf.constant(2)
b=tf.constant(3)

with tf.Session() as sess:
     print(sess.run(a),sess.run(b),sess.run(a+b),sess.run(a*b))

a=tf.placeholder(tf.int16)
b=tf.placeholder(tf.int16)
mul=tf.multiply(a,b)
add=tf.add(a,b)

with tf.Session() as sess:
     print(sess.run(add,feed_dict={a:2,b:3}))
     print(sess.run(mul,feed_dict={a:2,b:3}))

matrix1=tf.constant([[3.,3.]])
matrix2=tf.constant([[2.,2.]])
#since both are constants - need not send in feed_dict which happens if placeholders.
#dot product.
product=tf.matmul(matrix1,matrix2)
#multiplication 
multiply=tf.multiply(matrix1,matrix2)
with tf.Session() as sess:
     print(sess.run(matrix1),sess.run(matrix2),sess.run(matrix1+matrix2))
     print(sess.run(product))
     print(sess.run(multiply))

