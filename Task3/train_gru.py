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
multi_gru = []
for _ in range(layer_num):
    gru_cell = rnn.GRUCell(num_units=hidden_size)
    multi_gru.append(rnn.DropoutWrapper(cell=gru_cell, input_keep_prob=1.0, output_keep_prob=keep_prob))
mgru_cell = rnn.MultiRNNCell(multi_gru, state_is_tuple=True)
init_state = mgru_cell.zero_state(batch_size, dtype=tf.float32)

# outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
# outputs.shape = [batch_size, timestep_size, hidden_size]
outputs = []
cell_states = []
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = mgru_cell(X[:,timestep,:], state)
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
train_op = tf.train.RMSPropOptimizer(learning_rate=learningrate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', metavar='learning', type = float, default=0.1)
    parser.add_argument('-d', metavar='dropout', type = float, default=0.95)
    parser.add_argument('-b', metavar='batchsize', type = int, default=20)
    args = parser.parse_args()
    print "batchsize %d, dropout %.02f, learning %.02f" % (args.b, args.d, args.l)
    LOG_NAME = 'Log/train_b%d_d%.2f_l%.2f.log' % (args.b, args.d, args.l)
    with open(LOG_NAME, 'w') as f:
        print >> f, "Logging"        
    
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
    
    isTrain = 0
    if os.path.exists("save/model_b%d_d%.2f_l%.2f.ckpt.index" % (args.b, args.d, args.l)):
        isTrain = 1
        print "Restore model..."
        saver.restore(sess,"save/model_b%d_d%.2f_l%.2f.ckpt" % (args.b, args.d, args.l))
    
    max_accu = 0.0
    min_loss = 2
    max_train_accu = 0.0
    min_train_loss = 2
    turn = 0
    print "Training begin..."
    # g = train_batch_generator(_batch_size, timestep_size)
    g = train_batch_generator_sentence(_batch_size)
    for epoch in range(400):
        random.shuffle(trainingset)
        # for iter in range(train_size / (_batch_size * timestep_size)):
        for iter in range(train_sentence_size / _batch_size):
            batch = next(g)
            sess.run(train_op, feed_dict={_X:batch[0], y:batch[1], keep_prob:args.d, batch_size:_batch_size, learningrate:args.l})
        batch = next(g)
        train_loss,train_accu, trainpre = sess.run([cross_entropy, accuracy, y_pre], feed_dict={_X:batch[0], y:batch[1], keep_prob:1, batch_size:_batch_size})
        valid_loss,valid_accu, ypre = sess.run([cross_entropy, accuracy, y_pre], feed_dict={_X:valid_X, y:valid_y, keep_prob:1, batch_size:valid_batch})
        train_str = show_text(batch[1])
        trainpre_str = show_text(trainpre)
        true_str = show_text(valid_y)
        pre_str = show_text(ypre)
        with open(LOG_NAME, 'a') as f:
            print >> f, "train_str:%d\n%s" % (len(train_str.encode('utf-8')), train_str.encode('utf-8'))
            print >> f, "-------------------------------------\n"
            print >> f, "trainpre_str:%d\n%s" % (len(trainpre_str.encode('utf-8')), trainpre_str.encode('utf-8'))
            print >> f, "-------------------------------------\n"
            print >> f, "ytrue:%d\n%s" % (len(true_str.encode('utf-8')), true_str.encode('utf-8'))
            print >> f, "-------------------------------------\n"
            print >> f, "ypre:%d\n%s" % (len(pre_str.encode('utf-8')), pre_str.encode('utf-8'))
            print >> f, "-------------------------------------\n"
            print >> f, "epoch %d: train loss %f, train accu %f" % (epoch, train_loss, train_accu)
            print >> f, "epoch %d: valid loss %f, valid accu %f" % (epoch, valid_loss, valid_accu)
            print >> f, "-------------------------------------\n"
        print "epoch %d: train loss %f, train accu %f" % (epoch, train_loss, train_accu)
        print "epoch %d: valid loss %f, valid accu %f" % (epoch, valid_loss, valid_accu)
        if min_train_loss > train_loss:
            min_train_loss = train_loss
            saver_path = saver.save(sess,'save/train_loss_b%d_d%.2f_l%.2f.ckpt' % (args.b, args.d, args.l))
        elif max_train_accu < train_accu:
            max_train_accu = train_accu
            saver_path = saver.save(sess,'save/train_accu_b%d_d%.2f_l%.2f.ckpt' % (args.b, args.d, args.l))
        if min_loss > valid_loss:
            min_loss = valid_loss
            saver_path = saver.save(sess,'save/loss_b%d_d%.2f_l%.2f.ckpt' % (args.b, args.d, args.l))
            turn = 0
        if max_accu < valid_accu:
            max_accu = valid_accu
            saver_path = saver.save(sess,'save/accu_b%d_d%.2f_l%.2f.ckpt' % (args.b, args.d, args.l))
            turn = 0
        else:
            turn += 1
        print "min_loss:%f, max_accu:%f, turn:%d" % (min_loss, max_accu, turn) 
        # if turn == 10:
            # print "Early stop!"
            # break;
    
    print "Training End!"
    saver_path = saver.save(sess,"save/model_b%d_d%.2f_l%.2f.ckpt" % (args.b, args.d, args.l))
    
    with open(LOG_NAME, 'a') as f:
        print >> f, "****************************"
        print >> f, "Testing in validset begin..."
        saver.restore(sess,'save/model_b%d_d%.2f_l%.2f.ckpt' % (args.b, args.d, args.l))
        F = 1
        valid_loss = sess.run(cross_entropy, feed_dict={_X:valid_X, y:valid_y, keep_prob:1, batch_size:valid_batch})
        valid_accu = sess.run(accuracy, feed_dict={_X:valid_X, y:valid_y, keep_prob:1, batch_size:valid_batch})
        print >> f, "Loss %f, Accu %f" % (valid_loss, valid_accu)
        print >> f, "Testing End!"
