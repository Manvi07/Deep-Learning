#Script to load the values of trainable variables from a saved tensorflow model and save them to a numpy file.
#The given implementation is that for an hour glass.

import tensorflow as tf
import numpy as np
import network

with tf.Graph().as_default():
	phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
	pool3_placeholder = tf.placeholder(tf.float32, shape=(None, 40, 64, 128), name='pool3')
	hg_output = network.hg_inference(pool3_placeholder,  is_training = phase_train_placeholder, scope='hg')

	saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'hg'))

	a = []
	with tf.Session() as sess:
		saver.restore(sess, "./my-model-1000")

		for var in tf.trainable_variables():
			print(var)
			t = sess.run(var)
            print(t)
			a.append(t)

		a = np.array(a)
		print(a.shape)
		np.save('variables.npy', a)

print("Weights loaded")
exit()
