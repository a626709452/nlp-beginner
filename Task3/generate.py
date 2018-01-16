# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import pickle
import json
import argparse
import random

from tensorflow.contrib import rnn

from tools import *

config = tf.ConfigProto()
sess = tf.Session(config=config)
#sess = tf.InteractiveSession()

# init parameter #
# ########################################################### #
vocab_size = datasize['CharSize']

batch_size = tf.placeholder(tf.int32, [])

input_size = vocab_size

timestep_size = 25 # timestep

# hidden layer：64,128,256,512
hidden_size = 512

# LSTM layer：1,2,3
layer_num = 2

# learningrate
learningrate = tf.placeholder(tf.float32, [])

X = tf.placeholder(tf.float32, [None, vocab_size])
y = tf.placeholder(tf.float32, [None, vocab_size])
keep_prob = tf.placeholder(tf.float32, [])

# Build the model #
# ########################################################### #
# X = tf.reshape(_X, [-1, timestep_size, input_size])
multi_lstm = []
for _ in range(layer_num):
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    multi_lstm.append(rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob))
mlstm_cell = rnn.MultiRNNCell(multi_lstm, state_is_tuple=True)
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

W = tf.Variable(tf.random_uniform([hidden_size, vocab_size], minval=-0.08,maxval=0.08), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1, shape=[vocab_size]), dtype=tf.float32)

# outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
# outputs.shape = [batch_size, timestep_size, hidden_size]
outputs = []
cell_states = []
y_pre = []
y_p = []
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
            (cell_output, state) = mlstm_cell(y_p, state)
        else:
            (cell_output, state) = mlstm_cell(X, state)
        cell_states.append(state)
        y_prob = tf.nn.softmax(tf.matmul(cell_output, W) + bias) # [batch_size, class_size]
        y_p = np.zeros((1, vocab_size))
        y_p[0, np.argmax(y_prob)] = 1
        y_p = tf.convert_to_tensor(y_p, dtype=tf.float32)
        y_pre.append(y_prob)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', metavar=',model', type = int, choices=[1,2,3,4,5], 
                        help="1 for model, 2 for train_loss, 3 for train_accu, 4 for loss, 5 for accu", default=1)
    parser.add_argument('-l', metavar='learning', type = float, default=0.1)
    parser.add_argument('-d', metavar='dropout', type = float, default=0.95)
    parser.add_argument('-b', metavar='batchsize', type = int, default=20)
    parser.add_argument('-n', metavar='linenum', type = int, default=10)
    parser.add_argument('-o', metavar='output', default="output.txt")
    args = parser.parse_args()
    print "Total lines: %d, output file: %s" % (args.n, args.o)
    print "batchsize %d, dropout %.02f, learning %.02f" % (args.b, args.d, args.l)
            
    n = args.n
    
    if args.m == 1:
        model_name = "model"
    elif args.m == 2:
        model_name = "train_loss"
    elif agrs.m == 3:
        model_name = "train_accu"
    elif args.m == 4:
        model_name = "loss"
    elif agrs.m == 5:
        model_name = "accu"
    
    # X and y
    X_in = []
    X_seed = np.random.randint(0, datasize['CharSize'], n)
    for seed in X_seed:
        ch = ix_to_char[seed]
        while ch == u'，' or ch == u'。' or ch == u' ' or ch == u'\n':
            seed = np.random.randint(0, datasize['CharSize'])
            ch = ix_to_char[seed]
        x = np.zeros(datasize['CharSize'])
        x[seed] = 1
        X_in.append(x)
    
    if os.path.exists("save/%s_b%d_d%.2f_l%.2f.ckpt.index" % (model_name, args.b, args.d, args.l)):
        print "Restore model..."
        saver.restore(sess,"save/%s_b%d_d%.2f_l%.2f.ckpt" % (model_name, args.b, args.d, args.l))
    else:
        print "Check the model name"
        os.exit()

    print "Generate begin..."

    for i in range(n):
        ypre = sess.run(y_pre, feed_dict={X: X_in[i], keep_prob:1, batch_size:1})
        with open(args.o, 'w') as f:
            head = show_text[X_in[i]].encode('utf-8')
            pre_str = show_text[ypre[i]].encode('utf-8')
            print "%s%s" % (head, pre_str)
            print >> f, "%s%s" % (head, pre_str)
    
    print "Generate Finish!"