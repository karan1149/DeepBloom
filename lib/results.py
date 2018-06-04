from GRUModel import GRUModel
from DeepBloom import DeepBloom
from DeeperBloom import DeeperBloom
import json
from utils import *
from test_deep_bloom import test_gru_model
from test_bloom_filter import url_test

if __name__=='__main__':
	with open('../data/dataset.json', 'r') as f:
	    dataset = json.load(f)

	positives = dataset['positives']
	negatives = dataset['negatives']

	'''
	Baseline bloom filter 0.01 fp
	Bits needed 8020489
	Hash functions needed 6
	False positives 0.009922375051069045

	Baseline 0.001 fp
	Bits needed 12030733
	Hash functions needed 9
	False positives 0.0010336374778700803
	'''
	

	# positives = positives[:int(len(positives) * .05)]
	# negatives = negatives[:int(len(negatives) * .05)]
	print(len(positives), len(negatives))
	print("Baseline")
	url_test(positives, negatives, 0.01)
	# print("First attempt to get results, no data balancing")
	# model = GRUModel('../data/glove.6B.50d-char.txt', 50, pca_embedding_dim=16, maxlen=40, gru_size=16, batch_size=8192, lr=0.005, hidden_size=8, epochs=40)
	# test_gru_model(positives, negatives, model, train_dev_fraction=0.95, deeper_bloom=False, fp_rate=0.01)

	print("deeper bloom version with k = 2, dense only for second model")
	models = [
		GRUModel('../data/glove.6B.50d-char.txt', 50, pca_embedding_dim=16, maxlen=40, gru_size=16, batch_size=8192, lr=0.005, hidden_size=8, epochs=40), 
		GRUModel('../data/glove.6B.50d-char.txt', 50, pca_embedding_dim=16, maxlen=40, gru_size=16, batch_size=8192, lr=0.005, hidden_size=8, epochs=40, dense_only=True)
		]
	test_gru_model(positives, negatives, models, train_dev_fraction=0.95, deeper_bloom=True, fp_rate=0.01)

