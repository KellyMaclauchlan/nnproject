import tensorflow as tf
import numpy as np
import random
import downloadData
from sklearn.model_selection import train_test_split


#load the data and get the dictionary size
trX, trY, dictionarySize = downloadData.getData()

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(trX, trY, test_size=0.10)

#init weights
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

#the mlp
def model(X, w_h1, w_h2, w_h3, w_h4, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h1))   # layer 1
    h2 = tf.nn.relu(tf.matmul(h, w_h2))  # layer 2
    h3 = tf.nn.relu(tf.matmul(h2, w_h3)) # layer 3
    h4 = tf.nn.relu(tf.matmul(h3, w_h4)) # layer 4
    return tf.matmul(h4, w_o)

# relabel train and test
trX, teX,trY,  teY = X_train, X_test, y_train, y_test

#variables initilizaitons


size_h1 = tf.constant(3000, dtype=tf.int32)
size_h2 = tf.constant(3000, dtype=tf.int32)
size_h3 = tf.constant(3000, dtype=tf.int32)
size_h4 = tf.constant(3000, dtype=tf.int32)

X = tf.placeholder("float", [None, dictionarySize])
Y = tf.placeholder("float", [None, 2])


w_h1 = init_weights([dictionarySize, size_h1]) # create symbolic variables
w_h2 = init_weights([size_h1, size_h2])
w_h3 = init_weights([size_h2, size_h3])
w_h4 = init_weights([size_h3, size_h4])

w_o = init_weights([size_h4, 2])

py_x = model(X, w_h1,w_h2,w_h3,w_h4, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
	tf.global_variables_initializer().run()
	writer = tf.summary.FileWriter("project_graph")
	writer.add_graph(sess.graph)
    
	print(range(0,len(trX),128))
	for i in range(100):
		for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
			sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
		print(i, np.mean(np.argmax(teY, axis=1) ==
	                     sess.run(predict_op, feed_dict={X: teX})))


