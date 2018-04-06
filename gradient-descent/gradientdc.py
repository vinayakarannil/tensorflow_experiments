import tensorflow as tf
import numpy as np
import os,sys


x = np.asarray([8.0,11.0])
y = np.asarray([22.0,41.0])
n = x.shape[0]
print(n)

tf_X = tf.placeholder('float')
tf_Y = tf.placeholder('float')

a = tf.Variable(np.random.randn(), name='weights')
b = tf.Variable(np.random.randn(), name='bias')

prediction = tf.add(tf.multiply(a,tf_X),b)

loss= tf.reduce_sum(tf.pow((prediction-tf_Y),2))/(2*n)

optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)


init = tf.global_variables_initializer()

with tf.Session() as sess:

	sess.run(init)

	for epoch in range(10000):
		c,_ = sess.run([loss,optimizer], feed_dict={tf_X:x,tf_Y:y})
                print("Epoch:",epoch," || loss: ", c, "|| a :",sess.run(a), "|| b:", sess.run(b))
