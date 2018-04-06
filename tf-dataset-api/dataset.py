import tensorflow as tf
import numpy as np
import os,sys
from tensorflow.contrib.data import Dataset, Iterator
import matplotlib.pyplot as plt
import cv2
from PIL import Image
label_to_id = {'apparel':0,'footware':1}



def my_input_fn(img_path,label_name):
	img = tf.read_file(img_path)
	img_decode = tf.image.decode_jpeg(img, channels=0)
        images=tf.cast(img_decode, tf.float32)/255.0
	img_reshape = tf.image.resize_images(images,[300,300])
	return img_reshape,tf.one_hot(label_name,2)
	


images = []
labels = []

for i in os.listdir("data"):
	for j in os.listdir("data/"+i):
		images.append("data/"+i+"/"+j)
		labels.append(label_to_id[i])

img_list = tf.constant(images)
label_list = tf.constant(labels)


dataset = Dataset.from_tensor_slices((img_list,label_list)).map(my_input_fn)
dataset = dataset.batch(4)

#dataset = dataset.shuffle(buffer_size=2)

iterator = dataset.make_initializable_iterator()

next_element = iterator.get_next()

fig=plt.figure()

with tf.Session() as sess:

	sess.run(iterator.initializer)
	im = sess.run(next_element[0])
	print(im.shape)

	for i,image in enumerate(im):
		y = fig.add_subplot(2,2,i+1)
		y.imshow(im[i])
		
		plt.xticks([])
		plt.yticks([])
	plt.show();plt.pause(1);
	


	







