# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import pickle
import json

from tensorflow.contrib import rnn

from createdataset import *

# 设置 GPU 按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#sess = tf.InteractiveSession()

# Load input #
# ########################################################### #
Filename = "linux"
with open('cache/' + Filename + '_size.json') as f:
	size = json.load(f)
	vocab_size =size['vocab_size']
	data_size = size['train_size']
with open('cache/' + Filename + '_charix.pickle') as f:
	char_to_ix = pickle.load(f)
	ix_to_char = pickle.load(f)

# init parameter #
# ########################################################### #
batch_size = tf.placeholder(tf.int32, [])

# The num of char(1-of-k)
input_size = vocab_size
# timestep
timestep_size = 100

# hidden layer：64,128,256,512
hidden_size = 512
# LSTM layer：1,2,3
# layer_num = tf.placeholder(tf.int32, [])
layer_num = 3
# learningrate
learningrate = tf.placeholder(tf.float32, [])
# decay
decayrate = 0.95


_X = tf.placeholder(tf.float32, [None, vocab_size])
y = tf.placeholder(tf.float32, [None, vocab_size])
keep_prob = tf.placeholder(tf.float32, [])

# Build the model #
# ########################################################### #
X = tf.reshape(_X, [-1, timestep_size, input_size])
multi_lstm = []
for _ in range(layer_num):
	lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
	multi_lstm.append(rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob))
mlstm_cell = rnn.MultiRNNCell(multi_lstm, state_is_tuple=True)
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

# outputs.shape = [batch_size, timestep_size, hidden_size]
# outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
outputs = []
cell_states = []
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
		if timestep > 0:
			tf.get_variable_scope().reuse_variables()
		(cell_output, state) = mlstm_cell(X[:,timestep,:], state)
		cell_states.append(state)
		outputs.append(cell_output) # outputs.shape=[timestep_size, batch_size, hidden_size]
h_state = tf.reshape(tf.transpose(outputs, [1,0,2]), [-1, hidden_size])

# h_state = tf.reshape(outputs, [-1, hidden_size])


# train the model #
# ########################################################### #
W = tf.Variable(tf.random_uniform([hidden_size, vocab_size], minval=-0.08,maxval=0.08), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1, shape=[vocab_size]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)

# loss function
cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
train_op = tf.train.RMSPropOptimizer(learning_rate=learningrate).minimize(cross_entropy)
# train_op = tf.train.GradientDescentOptimizer(learningrate)

correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

isTrain = 0

print "Start Training!"
if os.path.exists('save/model.ckpt.meta'):
	isTrain = 1
	print "Restore model..."
	saver.restore(sess,'save/model.ckpt')
	
_batch_size = 100
batch_num = data_size / (_batch_size * timestep_size)
if batch_num * (_batch_size * timestep_size) >= data_size:
	batch_num -= 1
	
minloss = 2.0
maxacc = 0.2
flag = 0
epoch_num = 50
g = train_generator(Filename, batch_num, _batch_size, timestep_size)
_learning_rate = 2e-3
for epoch in range(epoch_num):
	for iter in range(batch_num):
		batch = next(g)
		if epoch < 10:
			sess.run(train_op, feed_dict={_X:batch[0], y:batch[1], keep_prob:1, batch_size:100, learningrate:_learning_rate})
		else:
			_learning_rate *= decayrate
			sess.run(train_op, feed_dict={_X:batch[0], y:batch[1], keep_prob:1, batch_size:100, learningrate:_learning_rate})
	#ytrue = ''.join([ix_to_char[ix] for ix in np.argmax(batch[1], 1)])
	temp = sess.run(y_pre, feed_dict={_X:batch[0], y:batch[1], keep_prob:1, batch_size:100})
	ypred = ''.join([ix_to_char[ix] for ix in np.argmax(temp, 1)])
	print "---ypred---\n%s\n" % (ypred)
	valid_batch_size, validbatch = Get_valid(Filename, timestep_size)
	train_accuracy = sess.run(accuracy, feed_dict={_X:batch[0], y:batch[1], keep_prob:1, batch_size:100})
	train_loss = sess.run(cross_entropy, feed_dict={_X:batch[0], y:batch[1], keep_prob:1, batch_size:100})
	print "Epoch %d, train loss %g, train accuracy %g" % ( epoch, train_loss, train_accuracy)
	valid_accuracy = sess.run(accuracy, feed_dict={_X:validbatch[0], y:validbatch[1], keep_prob:1, batch_size:valid_batch_size})
	valid_loss = sess.run(cross_entropy, feed_dict={_X:validbatch[0], y:validbatch[1], keep_prob:1, batch_size:valid_batch_size})
	print "Epoch %d, valid loss %g, valid accuracy %g" % ( epoch, valid_loss, valid_accuracy)
	if isTrain == 1 and epoch == 0:
		saver_path = saver.save(sess, 'save/model-retrain-epoch'+str(epoch)+'.ckpt')
		print "First retrainL, Model saved in file:", saver_path
	elif valid_loss < minloss:
		flag = 0
		minloss = valid_loss
		saver_path = saver.save(sess, 'save/model-loss-epoch'+str(epoch)+'.ckpt')
		print "Minloss update, Model saved in file:", saver_path
		if valid_accuracy > maxacc:
			maxacc = valid_accuracy
	elif valid_loss > minloss:
		flag += 1
	if flag > 4 or epoch == epoch_num - 1 :
		print "Training End!"
		saver_path = saver.save(sess, 'save/model.ckpt')
		print "Model saved in file:", saver_path
		break
	

sess.close()


