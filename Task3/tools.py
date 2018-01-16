# -*- coding:utf-8 -*-
import numpy as np
import os
import pickle
import json
import random

# Load input #
# ########################################################### #
with open('data/trainingset.txt') as f:
    trainingset = f.read().decode('utf-8').split('\n\n') # training set
with open('cache/data.pickle') as f:
    text = pickle.load(f)
    Poetry = pickle.load(f)
    char_to_ix = pickle.load(f)
    ix_to_char = pickle.load(f)
with open('cache/datasize.json') as f:
    datasize = json.load(f)
with open('data/devset.txt') as f:
    devset = f.read().decode('utf-8').split('\n\n') # dev set
with open('data/testset.txt') as f:
    testset = f.read().decode('utf-8').split('\n\n') # test set

def load_input():
    with open("data/poetryFromTang.txt") as f:
        text = f.read().decode('utf-8')

    CharVoc = list(set(text))
    CharSize = len(CharVoc)

    char_to_ix = { ch:i for i,ch in enumerate(CharVoc) }
    ix_to_char = { i:ch for i,ch in enumerate(CharVoc) }

    Poetry = text.split('\n\n')
    with open('cache/data.pickle','w') as f:
        pickle.dump(text, f)
        pickle.dump(Poetry, f)
        pickle.dump(char_to_ix, f)
        pickle.dump(ix_to_char, f)

    random.shuffle(Poetry)
    with open('data/trainingset.txt', 'w') as f:
        f.write('\n\n'.join(Poetry[:int(PoetrySize*0.9)]).encode('utf-8')) # training set
    with open('data/devset.txt', 'w') as f:
        f.write('\n\n'.join(Poetry[int(PoetrySize*0.9): int(PoetrySize*0.95)]).encode('utf-8')) # dev set
    with open('data/testset.txt', 'w') as f:
        f.write('\n\n'.join(Poetry[int(PoetrySize*0.95):]).encode('utf-8')) # test set


    PoetrySize = len(Poetry)
    PoetryEachLen = [len(p) for p in Poetry]

    datasize = dict()
    datasize['CharSize'] = CharSize
    datasize['PoetrySize'] = PoetrySize
    datasize['PoetryEachLen'] = PoetryEachLen

    with open('cache/datasize.json', 'w') as f:
        json.dump(datasize, f)

def train_batch_generator(batch_size, timestep_size):
    raw = '\n\n'.join(trainingset)
    data = [char_to_ix[ch] for ch in raw]
    class_name = datasize['CharSize']
    batch_num = batch_size * timestep_size
    iter = 0
    while 1:
        feature = []
        label = []
        for ch in data[iter * batch_num : (iter+1) * batch_num]:
            x = np.zeros(datasize['CharSize'])
            x[ch] = 1
            feature.append(x)
        label = feature[1:]
        nextbatchfirst = np.zeros(class_name)
        nextbatchfirst[data[(iter+1) * batch_num]] = 1
        label.append(nextbatchfirst)
        # with open("compare1.txt", 'w') as f:
            # print >> f, show_text(feature).encode('utf-8')
        # with open("compare2.txt", 'w') as f:
            # print >> f, show_text(label).encode('utf-8')
        yield [feature, label]
        iter += 1
        if (iter+1) * batch_size >= len(trainingset):
            iter = 0

def test_batch(type = 0):
    if type == 0:
        raw = '\n\n'.join(devset)
    else:
        raw = '\n\n'.join(testset)
    data = [char_to_ix[ch] for ch in raw]
    class_name = datasize['CharSize']
    feature = []
    label = []
    for ch in data[:-1]:
        x = np.zeros(datasize['CharSize'])
        x[ch] = 1
        feature.append(x)
    label = feature[1:]
    nextbatchfirst = np.zeros(class_name)
    nextbatchfirst[data[-1]] = 1
    label.append(nextbatchfirst)
    return [feature, label]
    
def train_batch_generator_sentence(batch_size):
    raw = '\n'.join(trainingset)
    raw_line = raw.split('\n')
    data = []
    for l in raw_line:
        if len(l) < datasize['LineMaxLen']:
            line = l + u' ' * (datasize['LineMaxLen'] - len(l)) 
        else:
            line = l
        data.append([char_to_ix[ch] for ch in line])
    class_name = datasize['CharSize']
    iter = 0
    while 1:
        feature = []
        label = []
        for line in data[iter * batch_size: (iter+1)*batch_size]:
            start = 1
            for ch in line:
                x = np.zeros(datasize['CharSize'])
                x[ch] = 1
                feature.append(x)
                if start != 1:
                    label.append(x)
                else:
                    start = 0
            endpoint = np.zeros(class_name)
            endpoint[char_to_ix[u'\n']] = 1
            label.append(endpoint)
        # with open("compare1.txt", 'w') as f:
            # print >> f, show_text(feature).encode('utf-8')
        # with open("compare2.txt", 'w') as f:
            # print >> f, show_text(label).encode('utf-8')
        yield [feature, label]
        iter += 1
        if (iter+1) * batch_size >= len(data):
            iter = 0

def test_batch_sentence(type = 0):
    if type == 0:
        raw = '\n'.join(devset)
    else:
        raw = '\n'.join(testset)
    raw_line = raw.split('\n')
    data = []
    for l in raw_line:
        if len(l) < datasize['LineMaxLen']:
            line = l + u' ' * (datasize['LineMaxLen'] - len(l)) 
        else:
            line = l
        data.append([char_to_ix[ch] for ch in line])
    class_name = datasize['CharSize']
    feature = []
    label = []
    for line in data[:-1]:
        start = 1
        for ch in line:
            x = np.zeros(datasize['CharSize'])
            x[ch] = 1
            feature.append(x)
            if start != 1:
                label.append(x)
            else:
                start = 0
        endpoint = np.zeros(class_name)
        endpoint[char_to_ix[u'\n']] = 1
        label.append(endpoint)
    return [feature, label]
    
def show_text(label):
    char_seq = [ix_to_char[np.argmax(ix)] for ix in label]
    return ''.join(char_seq)
