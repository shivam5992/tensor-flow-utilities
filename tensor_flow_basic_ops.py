import tensorflow as tf 

# Constant Operations 
c1 = tf.constant(10)
c2 = tf.constant(20)

# Default Graph (every operation is performed in a graph by default - Default Graph is selected)
# Every Graph is executed in a session
with tf.Session() as sess:
	print(sess.run(c1))
	print(sess.run(c2))
	print(sess.run(c1 - c2))
	print(sess.run(c1 + c2))

# Placeholders and Variables 
# In placeholders, the data is feeded (generally preferred for training examples)
# Variables - generally used for weights and coefficients
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
	print(sess.run(add, feed_dict = {a: 2, b: 4}))
	print(sess.run(mul, feed_dict = {a: 2, b: 4}))

## Matrix Multiplication
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
	print(sess.run(product))


