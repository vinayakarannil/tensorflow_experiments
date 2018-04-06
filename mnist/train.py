import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os
import random
import argparse

parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-train', required=True, 
                    help='Train or Predict')
arguments = parser.parse_args()
#Define parameters

checkpoint_dir = 'ckpt'
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
BATCH_SIZE = 128
LR = 0.001              # learning rate

#step1 : Read data
#using tensorflow's built in function. Take 2000 test images
mnist = input_data.read_data_sets('mnist', one_hot=True)  # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

#step2 : plot one example
#Plot the dimensions of MNIST dataset

print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape)   # (55000, 10)
plt.xticks([])
plt.yticks([])
number = random.randint(0, 20000)
plt.imshow(mnist.train.images[number].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[number])); plt.show()

#step3: create placeholders for features and labels
#each image in the MNIST is of size 28*28 = 784
#therefore each image is represented by a 784 tensor
#there are 10 classes for each image, corresponding to digit 0-9.
#each label is a one-hot vector

tf_x = tf.placeholder(tf.float32, [None, 28*28])
image = tf.reshape(tf_x, [-1, 28, 28, 1])              # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, 10])            # input y


#Step4: Build the model
#conv1==>pool1==>conv2==>pool2==>dense==>output
def model(image):
	with tf.name_scope("conv1"):
		conv1 = tf.layers.conv2d(   # shape (28, 28, 1)
		    inputs=image,
		    filters=32,
		    kernel_size=5,
		    strides=1,
		    padding='same',
		    activation=tf.nn.relu
		)           # -> (28, 28, 32)
	with tf.name_scope("pool1"):
		pool1 = tf.layers.max_pooling2d(
		    conv1,
		    pool_size=2,
		    strides=2,
		)           # -> (14, 14, 32)
	with tf.name_scope("conv2"):
		conv2 =tf.layers.conv2d(   # shape (14, 14, 32)
		    inputs=pool1,
		    filters=64,
		    kernel_size=5,
		    strides=1,
		    padding='same',
		    activation=tf.nn.relu
		)  

	with tf.name_scope("pool2"):
		pool2 = tf.layers.max_pooling2d(
		    conv2,
		    pool_size=2,
		    strides=2,
		)     # -> (7, 7, 64)
	flat = tf.reshape(pool2, [-1, 7*7*64])          # -> (7*7*32, )
	with tf.name_scope("dense1"):
        	dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
    	dropout = tf.layers.dropout(inputs=dense, rate=0.6)
       
    	with tf.name_scope("dense2"):
        	output = tf.layers.dense(dropout, units=10)              # output layer
    	return output

prediction = model(image)
#Step5: Define loss function
#use cross-entropy of softmax of logits as the loss function
#add loss to the summary to visualize in tensorboard
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=prediction) 

#step6 : Define training operation
#using Adam optimizer with learning rate of 0.001 to minimize loss
#if you want to use daying learning rate uncemment this section
##############################################################################
#global_step = tf.Variable(0, trainable=False)
#starter_learning_rate = 0.1
#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                           30, 0.99, staircase=True)
#tf.summary.scalar("learning rate",learning_rate)
#tf.summary.scalar("global step",global_step)
#train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
###################################################################################

train_op = tf.train.AdamOptimizer(LR).minimize(loss)

#Step7 : Check the accuracy of model using lables and predicted outputs

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(prediction, axis=1),)[1]

#Step8 : Add variables to tensorboard for tracking and visualisation

tf.summary.scalar("loss",loss)
tf.summary.histogram("loss",loss)
tf.summary.scalar("Accuracy",accuracy)
summary_op = tf.summary.merge_all()

#Step9 : Define a function to display some sample test results
def display_result(sess):
    test_output = sess.run(prediction, {tf_x: test_x[820:830]})
    pred_y = np.argmax(test_output, 1)
       
    for num,data in enumerate(test_x[820:830]):
        orig = data.reshape(28, 28)
        y = fig.add_subplot(2,5, num+1)
        y.imshow(data.reshape(28, 28),cmap='gray')
        plt.title(pred_y[num], fontsize=20)
        plt.xticks([])
        plt.yticks([])
    plt.show();plt.pause(1);

#stpe9 : Create tensorflow session
#initialize the variables
#run the training and save the checkpoints and graph for visualization

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

saver = tf.train.Saver()

plt.ion()

fig=plt.figure()

if(arguments.train == 'True'):
		
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	writer = tf.summary.FileWriter('ckpt/graphs', sess.graph)
	
        for step in range(600):
	    	b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
		
		
		_, loss_, summary = sess.run([train_op, loss, summary_op], {tf_x: b_x, tf_y: b_y})
	
			
		writer.add_summary(summary, step)
		
		fig.suptitle("STEP: %d" % step, fontsize=40)
                
		if step % 10 == 0:
			accuracy_ = sess.run(accuracy, {tf_x: test_x, tf_y: test_y})
			print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
			display_result(sess)
					
	
	
	save_path = saver.save(sess, checkpoint_prefix)
	writer.close() 
		
	plt.ioff()
	plt.close()

else:
	checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
	saver.restore(sess, checkpoint_file)
	display_result(sess)
	
	
	


