from GRUModel import GRUModel
from DeepBloom import DeepBloom
from DeeperBloom import DeeperBloom
import json
from utils import *
from test_deep_bloom import test_gru_model

if __name__=='__main__':
	with open('../data/dataset.json', 'r') as f:
	    dataset = json.load(f)

	positives = dataset['positives']
	negatives = dataset['negatives']

	print("First attempt to get results")
	model = GRUModel('../data/glove.6B.50d-char.txt', 50, pca_embedding_dim=None, maxlen=40, gru_size=16, batch_size=2048, lr=0.005, hidden_size=8)
	test_gru_model(positives, negatives, model, train_dev_fraction=0.95, deeper_bloom=False, fp_rate=0.01)

	print("deeper bloom version with k = 2")
	models = [GRUModel('../data/glove.6B.50d-char.txt', 50, pca_embedding_dim=None, maxlen=40, gru_size=16, batch_size=2048, lr=0.005, hidden_size=8) for _ in range(2)]
	test_gru_model(positives, negatives, models, train_dev_fraction=0.95, deeper_bloom=False, fp_rate=0.01)

