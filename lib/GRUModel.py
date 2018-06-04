from Model import Model
from utils import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten
from keras.layers import LSTM, Input, GRU
from keras import optimizers
from sklearn.decomposition import PCA
import numpy as np

class GRUModel(Model):
	def __init__(self, embeddings_path, embedding_dim, lr=0.001, maxlen=50, pca_embedding_dim=None, batch_size=1024, gru_size=16, hidden_size=None, second_gru_size=None, decay=0.0001, epochs=30, lstm=False, dense_only=False):
		self.embeddings_path = embeddings_path
		self.embedding_dim = embedding_dim
		self.lr = lr
		self.maxlen = maxlen
		self.pca_embedding_dim = pca_embedding_dim
		self.model = None
		self.batch_size = batch_size
		self.gru_size = gru_size
		self.hidden_size = hidden_size
		self.second_gru_size = second_gru_size
		self.decay = decay
		self.epochs = epochs
		self.lstm = lstm
		self.dense_only = dense_only

	def fit(self, text_X, text_y):

		X, y, self.char_indices, self.indices_char = vectorize_dataset(text_X, text_y, self.maxlen)
		num_chars = len(self.char_indices)

		embedding_vectors = {}
		with open(self.embeddings_path, 'r') as f:
		    for line in f:
		        line_split = line.strip().split(" ")
		        vec = np.array(line_split[1:], dtype=float)
		        char = line_split[0]
		        embedding_vectors[char] = vec

		embedding_matrix = np.zeros((num_chars + 1, self.embedding_dim))
		for char, i in self.char_indices.items():
		    embedding_vector = embedding_vectors.get(char)
		    assert(embedding_vector is not None)
		    embedding_matrix[i] = embedding_vector

		print(embedding_matrix.shape)

		if self.pca_embedding_dim:
		    pca = PCA(n_components=self.pca_embedding_dim)
		    pca.fit(embedding_matrix[1:])
		    embedding_matrix_pca = np.array(pca.transform(embedding_matrix[1:]))
		    embedding_matrix_pca = np.insert(embedding_matrix_pca, 0, 0, axis=0)
		    print("PCA matrix created")
		
		if not self.lstm:
			rnn_layer = GRU(self.gru_size, return_sequences=False if not self.second_gru_size else True)
		else:
			rnn_layer = LSTM(self.gru_size, return_sequences=False if not self.second_gru_size else True)


		prelayers = [
		    Embedding(num_chars + 1, self.embedding_dim if not self.pca_embedding_dim else self.pca_embedding_dim, input_length=self.maxlen,
    weights=[embedding_matrix] if not self.pca_embedding_dim else [embedding_matrix_pca]),
		    rnn_layer
		]

		if self.second_gru_size:
			prelayers.append(GRU(self.second_gru_size))

		if self.hidden_size:
			prelayers.append(Dense(self.hidden_size))

		postlayers = [
			Dense(1),
			Activation('sigmoid'),
		]

		if not self.dense_only:
			layers = prelayers + postlayers
		else:
			layers = [
		    Embedding(num_chars + 1, self.embedding_dim if not self.pca_embedding_dim else self.pca_embedding_dim, input_length=self.maxlen,
    weights=[embedding_matrix] if not self.pca_embedding_dim else [embedding_matrix_pca]),
		    Flatten(),
		    Dense(8),
		    Dense(4),
		    Dense(2),
		    Dense(1),
		    Activation('sigmoid'),
		]
	
		self.model = Sequential(layers)
		optimizer = optimizers.Adam(lr=self.lr, decay=self.decay)
		self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

		self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, verbose=2)
		# self.model.save("model.h5")
		# self.model = load_model('model.h5')


	def predict(self, text_x):
		x = np.zeros((1, self.maxlen), dtype=np.int)
		offset = max(self.maxlen - len(text_x), 0)
		for t, char in enumerate(text_x):
		    if t >= self.maxlen:
		        break
		    x[0, t + offset] = self.char_indices[char]
		pred = self.model.predict(x)
		return pred[0][0]

	# Like predict, but you pass in an array of URLs, and it is all
	# vectorized in one step, making it more efficient
	def predicts(self, text_X):
		X = np.zeros((len(text_X), self.maxlen), dtype=np.int)
		for i in range(len(text_X)):
			offset = max(self.maxlen - len(text_X[i]), 0)
			for t, char in enumerate(text_X[i]):
			    if t >= self.maxlen:
			        break
			    X[i, t + offset] = self.char_indices[char]
		preds = [pred[0] for pred in self.model.predict(X)]
		return preds
