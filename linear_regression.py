import tensorflow as tf 
import numpy 
import matplotlib.pyplot as plt 

rng = numpy.random

## Define parameters 

learning_rate = 0.01
training_epochs = 1000
display_step = 50 

## Training Data 
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
						 7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
						 2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]


## Create Graph 

# Input and Output Tensors 
X = tf.placeholder('float')
Y = tf.placeholder('float')

# Model Coefficients
W = tf.Variable(rng.randn(), name='weight')
b = tf.Variable(rng.randn(), name='bias')

# Prediction
y_pred = tf.add(tf.multiply(X, W), b)

# Mean Squared Error
cost = tf.reduce_mean(tf.pow(y_pred-Y, 2))/(2*n_samples)

# Gradient Descent 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Init Variables 
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	# Fit training Data
	for epoch in range(training_epochs):
		for (x, y) in zip(train_X, train_Y):

			# For each training example, run optimizer
			sess.run(optimizer, feed_dict={X:x, Y:y})

			# Update cost
			print(sess.run(cost, feed_dict={X:train_X, Y: train_Y}))

	final_cost = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
	print(final_cost)

	#Graphic display
	plt.plot(train_X, train_Y, 'ro', label='Original data')
	plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
	plt.legend()
	plt.show()