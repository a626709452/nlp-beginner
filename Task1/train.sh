#!/bin/sh
# python senti_classification.py -l 0.03
# python senti_classification.py -l 0.06
# python senti_classification.py -e 0
# python senti_classification.py -e 0.03

python senti_classification.py -f 2 -e 0.0
python senti_classification.py -f 3 -e 0.0
# feature 1 compare
python senti_classification.py -l 0.03 -e 0.0
python senti_classification.py -l 0.06 -e 0.0
python senti_classification.py -l 0.03 -e 0.03
python senti_classification.py -l 0.06 -e 0.03

# python senti_classification.py -f 2 -l 0.03
# python senti_classification.py -f 2 -l 0.06
# python senti_classification.py -f 2 -e 0
# python senti_classification.py -f 2 -e 0.03

# python senti_classification.py -f 3 -l 0.03
# python senti_classification.py -f 3 -l 0.06
# python senti_classification.py -f 3 -e 0
# python senti_classification.py -f 3 -e 0.03
