## Python TF script to create deep auto encoders and decoders 

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)



n_input = 784
n_hidden_1 = 256 
n_hidden_2 = 128
n_hidden_3 = 64 


weights = {
	'encoderh1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'encoderh2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'encoderh3' : tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
	'decoderh1' : tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
	'decoderh2' : tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
	'decoderh3' : tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}

bias = {
	'encoderh1' : tf.Variable(tf.random_normal([n_hidden_1])),
	'encoderh2' : tf.Variable(tf.random_normal([n_hidden_2])),
	'encoderh3' : tf.Variable(tf.random_normal([n_hidden_3])),
	'decoderh1' : tf.Variable(tf.random_normal([n_hidden_2])),
	'decoderh2' : tf.Variable(tf.random_normal([n_hidden_1])),
	'decoderh3' : tf.Variable(tf.random_normal([n_input])),
}


def _model(tr_x):

	# encoding part 

	z1 = tf.add(tf.matmul(tr_x, weights['encoderh1']), bias['encoderh1'])
	h1 = tf.nn.sigmoid(z1)

	z2 = tf.add(tf.matmul(h1, weights['encoderh2']), bias['encoderh2'])
	h2 = tf.nn.sigmoid(z2)

	z3 = tf.add(tf.matmul(h2, weights['encoderh3']), bias['encoderh3'])
	h3 = tf.nn.sigmoid(z3)

	# decoding part 
	
	z4 = tf.add(tf.matmul(h3, weights['decoderh1']), bias['decoderh1'])
	h4 = tf.nn.sigmoid(z4)

	z5 = tf.add(tf.matmul(h4, weights['decoderh2']), bias['decoderh2'])
	h5 = tf.nn.sigmoid(z5)

	z6 = tf.add(tf.matmul(h5, weights['decoderh3']), bias['decoderh3'])
	h6 = tf.nn.sigmoid(z6)

	return h6 


X = tf.placeholder("float", [None, n_input])

Y_pred = _model(X)
Y_actual = X 

## Training parameters 
learning_rate = 0.1 
batch_size = 500
training_epochs = 1

cost = tf.reduce_mean(tf.pow((Y_actual - Y_pred), 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

## Start training 
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

## start training 
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(training_epochs):
	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)

		o, c = sess.run([optimizer, cost], feed_dict={X:batch_xs})
		print i, c 


output = sess.run(Y_pred, feed_dict={X: mnist.test.images[:1]})

import numpy as np
import matplotlib.pyplot as plt

f, a = plt.subplots(2, 10, figsize=(10, 2))
for i, x in enumerate(output):
	a[0][0].imshow(np.reshape(mnist.test.images[i], (28, 28)))
	a[1][0].imshow(np.reshape(output[i], (28, 28)))
	if i == 9:
		break
f.savefig('f.png')
plt.draw()