from test_models import test_gru_model
import json

if __name__=='__main__':
	with open('../data/dataset.json', 'r') as f:
	    dataset = json.load(f)

	positives = dataset['positives']
	negatives = dataset['negatives']

	print("Data fraction 0.5, fp_rate 0.01, pca_embedding_dim 16, maxlen 50, gru size 16")
	test_gru_model(positives, negatives, data_fraction=0.5, fp_rate=0.01, pca_embedding_dim=16, maxlen=50, gru_size=16)

	print("Data fraction 0.5, fp_rate 0.01, pca_embedding_dim None, maxlen 50, gru size 16")
	test_gru_model(positives, negatives, data_fraction=0.5, fp_rate=0.01, pca_embedding_dim=None, maxlen=50, gru_size=16)