from __future__ import division
import numpy as np

class Model:
	def __init__(self, n, k, reg, learning):
		self.n = n
		self.k = k
		self.reg = reg
		self.learning = learning
		self.W = 0.01 * np.random.randn(n, k)
		self.b = np.zeros((1, k))
		

def eval_F(trueLabel, predLabel):
	labelset = set(trueLabel)
	eval = {}
	for l in labelset:
		tp = np.sum(predLabel[trueLabel == predLabel] == l)
		tn = np.sum(predLabel[trueLabel == predLabel] != l)
		fp = np.sum(predLabel[trueLabel != predLabel] == l)
		fn = np.sum(predLabel[trueLabel != predLabel] != l)
		precision = tp / (tp + fp)
		recall = tp / (tp + fn)
		f = 2 * precision * recall / (precision + recall)
		eval[l] = [precision, recall, f]
	return eval
	
def eval_accu(trueLabel, predLabel):
	tp = np.sum(trueLabel == predLabel)
	accuracy = tp / len(trueLabel)
	return accuracy