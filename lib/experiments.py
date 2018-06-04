from test_models import test_gru_model
import json

if __name__=='__main__':
	with open('../data/dataset.json', 'r') as f:
	    dataset = json.load(f)

	positives = dataset['positives']
	negatives = dataset['negatives']


	print("Data fraction 0.3, fp_rate 0.005, pca_embedding_dim 16, maxlen 40, gru size 16, batch size 1024, lr 0.005, 40 epochs, hidden size 8")
	test_gru_model(positives, negatives, data_fraction=0.3, fp_rate=0.005, pca_embedding_dim=16, maxlen=40, gru_size=16, batch_size=1024, lr=0.005, hidden_size=8, epochs=40)

	print("Data fraction 0.3, fp_rate 0.005, pca_embedding_dim 16, maxlen 40, gru size 16, batch size 1024, lr 0.005, 40 epochs, hidden size None")
	test_gru_model(positives, negatives, data_fraction=0.3, fp_rate=0.005, pca_embedding_dim=16, maxlen=40, gru_size=16, batch_size=1024, lr=0.005, hidden_size=None, epochs=40)

	print("Data fraction 0.3, fp_rate 0.005, pca_embedding_dim 16, maxlen 40, gru size 16, batch size 1024, lr 0.005, 40 epochs, hidden size 8, decay is 0")
	test_gru_model(positives, negatives, data_fraction=0.3, fp_rate=0.005, pca_embedding_dim=16, maxlen=40, gru_size=16, batch_size=1024, lr=0.005, hidden_size=8, decay=0.0, epochs=40)

	'''
	Using threshold 0.5
	0.12742250957053114 false negatives for positives set.
	0.15276453765490944 false positive rate for train.
	0.15897226383403695 false positive rate for dev.
	0.15683871260611013 false positive rate for test.
	Getting threshold for fp_rate 0.005
	Using threshold 0.9795848
	0.5072759938015624 false negatives for positives set.
	0.0046245857732988335 false positive rate for train.
	0.004948023060511144 false positive rate for dev.
	0.005765127786100141 false positive rate for test.
	'''
	# print("Data fraction 0.3, fp_rate 0.005, pca_embedding_dim None, maxlen 40, gru size 16, batch size 2048, lr 0.005, 30 epochs, hidden size 8")
	# test_gru_model(positives, negatives, data_fraction=0.3, fp_rate=0.005, pca_embedding_dim=None, maxlen=40, gru_size=16, batch_size=2048, lr=0.005, hidden_size=8)

	'''
	Using threshold 0.5
	0.13347355505893693 false negatives for positives set.
	0.12320690907440192 false positive rate for train.
	0.12978346724771891 false positive rate for dev.
	0.13023741431749059 false positive rate for test.
	Getting threshold for fp_rate 0.005
	Using threshold 0.976086
	0.4669941162645251 false negatives for positives set.
	0.004874256661673249 false positive rate for train.
	0.004948023060511144 false positive rate for dev.
	0.006582232511689137 false positive rate for test.
	'''
	# print("Data fraction 0.3, fp_rate 0.005, pca_embedding_dim None, maxlen 40, gru size 16, batch size 2048, lr 0.005, 30 epochs, second gru size 8")
	# test_gru_model(positives, negatives, data_fraction=0.3, fp_rate=0.005, pca_embedding_dim=None, maxlen=40, gru_size=16, batch_size=2048, lr=0.005, second_gru_size=8)

	'''
	Using threshold 0.5
	0.12800809461779622 false negatives for positives set.
	0.1345158654500885 false positive rate for train.
	0.14249398520132553 false positive rate for dev.
	0.14263016932225703 false positive rate for test.
	Getting threshold for fp_rate 0.005
	Using threshold 0.9783211
	0.48746170791655213 false negatives for positives set.
	0.00407417495120069 false positive rate for train.
	0.004948023060511144 false positive rate for dev.
	0.004312497162830814 false positive rate for test.
	'''
	# print("Data fraction 0.3, fp_rate 0.005, pca_embedding_dim None, maxlen 40, gru size 16, batch size 2048, lr 0.005, 30 epochs, second gru size 8, hidden size 4")
	# test_gru_model(positives, negatives, data_fraction=0.3, fp_rate=0.005, pca_embedding_dim=None, maxlen=40, gru_size=16, batch_size=2048, lr=0.005, second_gru_size=8, hidden_size=4)


	'''
	Params 5183
	Using threshold 0.5
	0.1689791300676012 false negatives for positives set.
	0.15416609923282945 false positive rate for train.
	0.1613327885968496 false positive rate for dev.
	0.1563393708293613 false positive rate for test.
	Getting threshold for fp_rate 0.005
	Using threshold 0.9729382
	0.6073313654488888 false negatives for positives set.
	0.004528122020972355 false positive rate for train.
	0.004948023060511144 false positive rate for dev.
	0.0044486812837623135 false positive rate for test.
	'''
	# print("Data fraction 0.3, fp_rate 0.005, pca_embedding_dim None, maxlen 20, gru size 16, batch size 2048, lr 0.001, 30 epochs")
	# test_gru_model(positives, negatives, data_fraction=0.3, fp_rate=0.005, pca_embedding_dim=None, maxlen=20, gru_size=16, batch_size=2048)


	'''
	Params 5183
	Using threshold 0.5
	0.14549199102899643 false negatives for positives set.
	0.17686912705978483 false positive rate for train.
	0.18657224567615416 false positive rate for dev.
	0.1820327749784375 false positive rate for test.
	Getting threshold for fp_rate 0.005
	Using threshold 0.980731
	0.6254446661966052 false negatives for positives set.
	0.004289799809342231 false positive rate for train.
	0.004948023060511144 false positive rate for dev.
	0.00449407599073948 false positive rate for test.
	'''
	# print("Data fraction 0.3, fp_rate 0.005, pca_embedding_dim None, maxlen 30, gru size 16, batch size 2048, lr 0.001, 30 epochs")
	# test_gru_model(positives, negatives, data_fraction=0.3, fp_rate=0.005, pca_embedding_dim=None, maxlen=30, gru_size=16, batch_size=2048)


	'''
	Params 5183
	Using threshold 0.5
	0.14604969107401078 false negatives for positives set.
	0.17324889917835581 false positive rate for train.
	0.1784011984202642 false positive rate for dev.
	0.1778564619365382 false positive rate for test.
	Getting threshold for fp_rate 0.005
	Using threshold 0.9824494
	0.6140875031370627 false negatives for positives set.
	0.004647283126787417 false positive rate for train.
	0.004948023060511144 false positive rate for dev.
	0.004766444232602479 false positive rate for test.
	'''
	# print("Data fraction 0.3, fp_rate 0.005, pca_embedding_dim None, maxlen 40, gru size 16, batch size 2048, lr 0.001, 30 epochs")
	# test_gru_model(positives, negatives, data_fraction=0.3, fp_rate=0.005, pca_embedding_dim=None, maxlen=40, gru_size=16, batch_size=2048)


	'''
	Params 5183
	Using threshold 0.5
	0.18301325334321258 false negatives for positives set.
	0.14209678151527533 false positive rate for train.
	0.1464887194153162 false positive rate for dev.
	0.14571700939670434 false positive rate for test.
	Getting threshold for fp_rate 0.005
	Using threshold 0.9720799
	0.6216642566057579 false negatives for positives set.
	0.004709700848881021 false positive rate for train.
	0.004948023060511144 false positive rate for dev.
	0.0046302601116709795 false positive rate for test.
	'''
	# print("Data fraction 0.3, fp_rate 0.005, pca_embedding_dim None, maxlen 50, gru size 16, batch size 2048, lr 0.001, 30 epochs")
	# test_gru_model(positives, negatives, data_fraction=0.3, fp_rate=0.005, pca_embedding_dim=None, maxlen=50, gru_size=16, batch_size=2048)


	'''
	Using threshold 0.5
	0.1504356035708737 false negatives for positives set.
	0.17470152980162512 false positive rate for train.
	0.1803077761133052 false positive rate for dev.
	0.17921830314585319 false positive rate for test.
	Getting threshold for fp_rate 0.005
	Using threshold 0.9776559
	0.6065107496683676 false negatives for positives set.
	0.004783467247718916 false positive rate for train.
	0.004948023060511144 false positive rate for dev.
	0.004811838939579645 false positive rate for test.
	'''
	# print("Data fraction 0.3, fp_rate 0.005, pca_embedding_dim None, maxlen 40, gru size 16, batch size 2048, lr 0.001, 30 epochs")
	# test_gru_model(positives, negatives, data_fraction=0.3, fp_rate=0.005, pca_embedding_dim=None, maxlen=40, gru_size=16, batch_size=2048)


	'''
	Using threshold 0.5
	0.14940784205934726 false negatives for positives set.
	0.11417903672431795 false positive rate for train.
	0.12224794588950928 false positive rate for dev.
	0.11988742112669663 false positive rate for test.
	Getting threshold for fp_rate 0.005
	Using threshold 0.9669876
	0.5158964430687843 false negatives for positives set.
	0.004647283126787417 false positive rate for train.
	0.004948023060511144 false positive rate for dev.
	0.0052203913023741435 false positive rate for test.
	'''
	# print("Data fraction 0.3, fp_rate 0.005, pca_embedding_dim None, maxlen 40, gru size 16, batch size 2048, lr 0.005, 30 epochs")
	# test_gru_model(positives, negatives, data_fraction=0.3, fp_rate=0.005, pca_embedding_dim=None, maxlen=40, gru_size=16, batch_size=2048, lr=0.005)


	'''
	Using threshold 0.5
	0.17042915018463856 false negatives for positives set.
	0.09253711017295384 false positive rate for train.
	0.09678151527531889 false positive rate for dev.
	0.09977756593581188 false positive rate for test.
	Getting threshold for fp_rate 0.005
	Using threshold 0.9676666
	0.4923814190279288 false negatives for positives set.
	0.004721049525625312 false positive rate for train.
	0.004948023060511144 false positive rate for dev.
	0.005583548958191475 false positive rate for test.
	'''
	# print("Data fraction 0.3, fp_rate 0.005, pca_embedding_dim None, maxlen 40, gru size 16, batch size 2048, lr 0.01, 30 epochs")
	# test_gru_model(positives, negatives, data_fraction=0.3, fp_rate=0.005, pca_embedding_dim=None, maxlen=40, gru_size=16, batch_size=2048, lr=0.01)

	'''
	Using threshold 0.5
	0.23638514765108692 false negatives for positives set.
	0.2864519497026647 false positive rate for train.
	0.2903899405329339 false positive rate for dev.
	0.2906169140678197 false positive rate for test.
	Getting threshold for fp_rate 0.005
	Using threshold 0.96612453
	0.809828268221853 false negatives for positives set.
	0.0048629079849289575 false positive rate for train.
	0.004948023060511144 false positive rate for dev.
	0.004811838939579645 false positive rate for test.
	'''
	# print("Data fraction 0.3, fp_rate 0.005, pca_embedding_dim None, maxlen 40, gru size 16, batch size 2048, lr 1e-4, 30 epochs")
	# test_gru_model(positives, negatives, data_fraction=0.3, fp_rate=0.005, pca_embedding_dim=None, maxlen=40, gru_size=16, batch_size=2048, lr=1e-4)

	
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
	# print("Data fraction 0.5, fp_rate 0.01, pca_embedding_dim 16, maxlen 50, gru size 16, batch size 2048, lr 0.001, 30 epochs")
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
	# print("Data fraction 0.5, fp_rate 0.01, pca_embedding_dim None, maxlen 50, gru size 16, batch size 2048, lr 0.001, 30 epochs")
	# test_gru_model(positives, negatives, data_fraction=0.5, fp_rate=0.01, pca_embedding_dim=None, maxlen=50, gru_size=16, batch_size=2048)