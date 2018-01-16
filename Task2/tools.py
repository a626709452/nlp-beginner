from __future__ import division
import numpy as np
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import pickle

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
    
# input:
#	trainset: zip tuple
# 	feature: int
def get_voc(trainset, feature):
	phrase, sentiment = zip(*trainset)
	text = ' '.join(phrase)
	stop = set(stopwords.words('english'))
	Voc = []
	if feature == 1:
		fdist = nltk.FreqDist(text.split(' '))
		sorted_voc = fdist.most_common()
		voc_without_stop = [w for w,_ in sorted_voc if w not in stop]
		Voc = voc_without_stop[:10000]
	elif feature == 2:
		bigrams = []
		for i in range(len(phrase)):
			origin = list(nltk.bigrams(phrase[i].split(' ')))
			clean = [(a, b) for (a, b) in origin if a.isalpha() and b.isalpha()
					and (a not in stop or b not in stop)]
			bigrams.extend(clean)
		fdist = nltk.FreqDist(bigrams)
		sorted_bigram = fdist.most_common()
		bigram_clean = [bi for bi,_ in sorted_bigram]
		Voc = bigram_clean[:30000]
	elif feature == 3:
		trigrams = []
		for i in range(len(phrase)):
			origin = list(nltk.trigrams(phrase[i].split(' ')))
			clean = [(a, b, c) for (a, b, c) in origin if 
						a.isalpha() and b.isalpha() and c.isalpha() and 
						(a not in stop or b not in stop or c not in stop)]
			trigrams.extend(clean)
		fdist = nltk.FreqDist(trigrams)
		sorted_trigram = fdist.most_common()
		trigram_clean = [bi for bi,_ in sorted_trigram]
		Voc = trigram_clean[:50000]
	else:
		print "Unkown feature type!"
		sys.exit(1)
	ngram_to_ix = { ch:i for i,ch in enumerate(Voc) }
	return ngram_to_ix
    
    
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', metavar='feature', type = int, choices=[1,2,3,4], help="1 for bag-of-word, 2 for bigram, 3 for trigram, 4 for word2vec", default=1)
	parser.add_argument('-l', metavar='learning', type = float, default=0.01)
	parser.add_argument('-e', metavar='reg', type = float, default=0.01)
	args = parser.parse_args()
	print "feature %d, learning %.02f, reg %.02f" % (args.f, args.l, args.e)
	LOG_NAME = 'train_f%d_l%.2f_e%.2f.log' % (args.f, args.l, args.e)
	with open(LOG_NAME, 'w') as f:
		print >> f, "Logging"
	
	trainset, validset = load_train()
	print "trainset length: %d, validset length: %d" % (len(trainset), len(validset))
	
	if(args.f < 4):
		print "Generating Voc..."
		if not os.path.exists("cache/ngram_f%d.pickle" % args.f):
			print "  Saving new ngram_to_ix..."
			ngram_to_ix = get_voc(trainset, args.f)
			with open("cache/ngram_f%d.pickle" % args.f, 'w') as f:
				pickle.dump(ngram_to_ix, f)
		else:
			print "  Loading original ngram_to_ix..."
			with open("cache/ngram_f%d.pickle" % args.f, 'r') as f:
				ngram_to_ix = pickle.load(f)
		n = len(ngram_to_ix)
	else:
		print "Generating Word2Vec..."
		n_dim = 500
		if not os.path.exists('w2v/imdb_w2v'):
			print "  Saving new Word2Vec..."
			phrase, sentiment = zip(*trainset)
			x_train = []
			for i in range(len(phrase)):
				x_train.append(phrase[i].lower().split())
			imdb_w2v = Word2Vec(x_train, size=n_dim, min_count=1)
			imdb_w2v.save("w2v/imdb_w2v")
			imdb_w2v.wv.save_word2vec_format("w2v/word2vec_org",
											  "w2v/vocabulary",
											  binary=False)
		else:
			print "  Loading original Word2Vec..."
			imdb_w2v = Word2Vec.load('w2v/imdb_w2v')
		ngram_to_ix = (n_dim, imdb_w2v)
		n = n_dim