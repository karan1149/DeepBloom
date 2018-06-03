from test_models import test_gru_model
import json

if __name__=='__main__':
	with open('../data/dataset.json', 'r') as f:
	    dataset = json.load(f)

	positives = dataset['positives']
	negatives = dataset['negatives']



	print("Data fraction 0.5, fp_rate 0.005, pca_embedding_dim None, maxlen 30, gru size 16, batch size 2048")
	test_gru_model(positives, negatives, data_fraction=0.5, fp_rate=0.01, pca_embedding_dim=None, maxlen=30, gru_size=16, batch_size=2048)

	print("Data fraction 0.5, fp_rate 0.005, pca_embedding_dim None, maxlen 40, gru size 16, batch size 2048")
	test_gru_model(positives, negatives, data_fraction=0.5, fp_rate=0.01, pca_embedding_dim=None, maxlen=40, gru_size=16, batch_size=2048)

	print("Data fraction 0.5, fp_rate 0.005, pca_embedding_dim None, maxlen 50, gru size 16, batch size 2048")
	test_gru_model(positives, negatives, data_fraction=0.5, fp_rate=0.01, pca_embedding_dim=None, maxlen=50, gru_size=16, batch_size=2048)

	print("Data fraction 0.5, fp_rate 0.005, pca_embedding_dim None, maxlen 60, gru size 16, batch size 2048")
	test_gru_model(positives, negatives, data_fraction=0.5, fp_rate=0.01, pca_embedding_dim=None, maxlen=60, gru_size=16, batch_size=2048)
	
	'''
	Using threshold 0.5
	0.17059646019814287 false negatives for positives set.
	0.1496697535067411 false positive rate for train.
	0.1501021380906986 false positive rate for dev.
	0.14993871714558082 false positive rate for test.
	Getting threshold for fp_rate 0.01
	Using threshold 0.9496127
	0.48878903402368634 false negatives for positives set.
	0.009849516546370693 false positive rate for train.
	0.009968677652185755 false positive rate for dev.
	0.009832493531254255 false positive rate for test.
	'''
	# print("Data fraction 0.5, fp_rate 0.01, pca_embedding_dim 16, maxlen 50, gru size 16, batch size 2048")
	# test_gru_model(positives, negatives, data_fraction=0.5, fp_rate=0.01, pca_embedding_dim=16, maxlen=50, gru_size=16, batch_size=2048)

	

	'''
	Using threshold 0.5
	0.15557441112850603 false negatives for positives set.
	0.15458940487539152 false positive rate for train.
	0.1585455535884516 false positive rate for dev.
	0.1546506877298107 false positive rate for test.
	Getting threshold for fp_rate 0.01
	Using threshold 0.9549814
	0.4746513378825723 false negatives for positives set.
	0.009573743701484406 false positive rate for train.
	0.009968677652185755 false positive rate for dev.
	0.009560125289391257 false positive rate for test.
	'''
	# print("Data fraction 0.5, fp_rate 0.01, pca_embedding_dim None, maxlen 50, gru size 16, batch size 2048")
	# test_gru_model(positives, negatives, data_fraction=0.5, fp_rate=0.01, pca_embedding_dim=None, maxlen=50, gru_size=16, batch_size=2048)