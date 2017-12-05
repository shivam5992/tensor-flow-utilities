import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

## Define parameters 

learning_rate = 0.01
training_epochs = 100
batch_size = 100
display_step = 1

## Create Graph 

# Input and Output Tensors 
# !! Dimensions - None, 784
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Model Coefficients
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Prediction
y_pred = tf.nn.softmax(tf.matmul(x, W)+ b)

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_pred), reduction_indices=1))

# Gradient Descent 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Init Variables 
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	# Fit training Data

	## Non Stochastic Gradient Descent 
	for epoch in range(training_epochs):
		avg_cost = 0
		total_batch = int(mnist.train.num_examples/batch_size)
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)

			print batch_ys
			exit(0)


			_, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
														  y: batch_ys})
			# Compute average loss
			avg_cost += c / total_batch


	# Test model
	correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))

	# Calculate accuracy for 3000 examples
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print (accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))