# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import pickle
import json
import argparse

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

# outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
# outputs.shape = [batch_size, timestep_size, hidden_size]
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

# train the model #
# ########################################################### #
W = tf.Variable(tf.random_uniform([hidden_size, vocab_size], minval=-0.08,maxval=0.08), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1, shape=[vocab_size]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)

# loss function
cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))

correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', metavar=',model', type = int, choices=[1,2,3], help="1 for model, 2 for loss, 3 for accu", default=1)
    parser.add_argument('-l', metavar='learning', type = float, default=0.1)
    parser.add_argument('-d', metavar='dropout', type = float, default=0.95)
    parser.add_argument('-b', metavar='batchsize', type = int, default=20)
    args = parser.parse_args()
    print "batchsize %d, dropout %.02f, learning %.02f" % (args.b, args.d, args.l)
    
    # X and y
    raw = '\n\n'.join(trainingset)
    train_size = len(raw)
    valid_X, valid_y = test_batch_sentence()
    valid_batch = len(valid_y) / timestep_size
    valid_X = valid_X[:valid_batch * timestep_size]
    valid_y = valid_y[:valid_batch * timestep_size]
    test_X, test_y = test_batch_sentence(type=1)
    test_batch = len(test_y) / timestep_size
    test_X = test_X[:test_batch * timestep_size]
    test_y = test_y[:test_batch * timestep_size]
    train_sentence_size = len(('\n'.join(trainingset)).split('\n'))
    print "train_sentence_size %d, valid_batch %d, test_batch %d" % (train_sentence_size, valid_batch, test_batch)
    
    
    # model parameters
    _batch_size = args.b
    if args.m == 1:
        model_name = "model"
    elif args.m == 2:
        model_name = "loss"
    else:
        model_name = "accu"
        
    if os.path.exists("save/%s_b%d_d%.2f_l%.2f.ckpt.index" % (model_name, args.b, args.d, args.l)):
        print "Restore model..."
        print "save/%s_b%d_d%.2f_l%.2f.ckpt" % (model_name, args.b, args.d, args.l)
        saver.restore(sess,"save/%s_b%d_d%.2f_l%.2f.ckpt" % (model_name, args.b, args.d, args.l))
    else:
        print "Check the model name"
        exit()
    
    LOG_NAME = 'Log/test_%s_b%d_d%.2f_l%.2f.log' % (model_name, args.b, args.d, args.l)
    with open(LOG_NAME, 'w') as f:
        print >> f, "Logging"        

    print "Test begin..."
    g = train_batch_generator_sentence(_batch_size)
    
    print "Test the training set"
    cross_entropy_sum = 0
    accuracy_sum = 0
    for i in range(3):
        cross_entropy_all = []
        accuracy_all = []
        for iter in range(train_sentence_size / _batch_size):
            batch = next(g)
            train_loss,train_accu, trainpre = sess.run([cross_entropy, accuracy, y_pre], feed_dict={_X:batch[0], y:batch[1], keep_prob:1, batch_size:_batch_size})
            cross_entropy_all.append(train_loss)
            accuracy_all.append(train_accu)
            
            train_str = show_text(batch[1])
            trainpre_str = show_text(trainpre)
            with open(LOG_NAME, 'a') as f:
                print >> f, "\nTest %d, batch %d" % (i, iter)
                print >> f, "train_str:%d\n%s" % (len(train_str.encode('utf-8')), train_str.encode('utf-8'))
                print >> f, "-------------------------------------\n"
                print >> f, "trainpre_str:%d\n%s" % (len(trainpre_str.encode('utf-8')), trainpre_str.encode('utf-8'))
                print >> f, "-------------------------------------\n"
            
        cross_entropy_sum += np.mean(cross_entropy_all)
        accuracy_sum += np.mean(accuracy_all)
    
        with open(LOG_NAME, 'a') as f:
            print >> f, "train loss %f,  train accu %f" % (np.mean(cross_entropy_all), np.mean(accuracy_all))
    with open(LOG_NAME, 'a') as f:
        print >> f, "avg train loss %f, avg train accu %f" % (cross_entropy_sum / 3, accuracy_sum / 3)
        
        
    print "Test the valid set"
    cross_entropy_sum = 0
    accuracy_sum = 0
    for i in range(3):
        valid_loss,valid_accu, ypre = sess.run([cross_entropy, accuracy, y_pre], feed_dict={_X:valid_X, y:valid_y, keep_prob:1, batch_size:valid_batch})
        true_str = show_text(valid_y)
        pre_str = show_text(ypre)
        with open(LOG_NAME, 'a') as f:
            print >> f, "\nTest %d" % (i)
            print >> f, "ytrue:%d\n%s" % (len(true_str.encode('utf-8')), true_str.encode('utf-8'))
            print >> f, "-------------------------------------\n"
            print >> f, "ypre:%d\n%s" % (len(pre_str.encode('utf-8')), pre_str.encode('utf-8'))
            print >> f, "-------------------------------------\n"
            
        with open(LOG_NAME, 'a') as f:
            print >> f, "valid loss %f,  valid accu %f" % (valid_loss, valid_accu)
        
        cross_entropy_sum += valid_loss
        accuracy_sum += valid_accu

    with open(LOG_NAME, 'a') as f:
        print >> f, "avg valid loss %f, avg test accu %f" % (cross_entropy_sum / 3, accuracy_sum / 3)
    
    
    # print "Test the test set"
    # cross_entropy_sum = 0
    # accuracy_sum = 0
    # for i in range(3):
        # test_loss,test_accu, ypre = sess.run([cross_entropy, accuracy, y_pre], feed_dict={_X:test_X, y:test_y, keep_prob:1, batch_size:test_batch})
        # true_str = show_text(test_y)
        # pre_str = show_text(ypre)
        # with open(LOG_NAME, 'a') as f:
            # print >> f, "Test %d\n" % (i)
            # print >> f, "ytrue:%d\n%s" % (len(true_str.encode('utf-8')), true_str.encode('utf-8'))
            # print >> f, "-------------------------------------\n"
            # print >> f, "ypre:%d\n%s" % (len(pre_str.encode('utf-8')), pre_str.encode('utf-8'))
            # print >> f, "-------------------------------------\n"
            
        # with open(LOG_NAME, 'a') as f:
            # print >> f, "test loss %f,  test accu %f" % (test_loss, test_accu)
        
        # cross_entropy_sum += test_loss
        # accuracy_sum += test_accu

    # with open(LOG_NAME, 'a') as f:
        # print >> f, "avg test loss %f, avg test accu %f" % (cross_entropy_sum / 3, accuracy_sum / 3)
        
    print "Test End"