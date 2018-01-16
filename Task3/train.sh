#!/bin/sh
python train_gru.py -l 0.1 -d 0.9 -b 20
python train_gru.py -l 0.06 -d 0.9 -b 20
python train_gru.py -l 0.03 -d 0.9 -b 20

python train_gru.py -l 0.03 -d 0.9 -b 30
python train_gru.py -l 0.03 -d 0.9 -b 40

python train_gru.py -l 0.1 -d 0.85 -b 20
python train_gru.py -l 0.1 -d 0.95 -b 20

python test_gru.py -l 0.1 -d 0.9 -b 20
python test_gru.py -l 0.06 -d 0.9 -b 20
python test_gru.py -l 0.03 -d 0.9 -b 20

python test_gru.py -l 0.03 -d 0.9 -b 30
python test_gru.py -l 0.03 -d 0.9 -b 40

python test_gru.py -l 0.1 -d 0.85 -b 20
python test_gru.py -l 0.1 -d 0.95 -b 20
