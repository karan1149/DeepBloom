from Model import Model
from utils import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, Input, GRU
from keras import optimizers

class GRUModel(Model):
	def __init__(self, embeddings_path, embedding_dim, lr, maxlen=50):
		self.embeddings_path = embeddings_path
		self.embedding_dim = embedding_dim
		self.lr = lr
		self.maxlen = maxlen
		self.model = None

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
		    
	
		self.model = Sequential([
		    Embedding(num_chars + 1, self.embedding_dim, input_length=self.maxlen,
    weights=[embedding_matrix]),
		    GRU(16),
		    Dense(1),
		    Activation('sigmoid'),
		])
		optimizer = optimizers.Adam(lr=self.lr, decay=0.0001)
		self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

		self.model.fit(X, y, batch_size=128, epochs=10)
		self.model.save("model.h5")
		# self.model = load_model('model.h5')


	def predict(self, text_x):
		x = np.zeros((1, self.maxlen), dtype=np.int)
		offset = max(self.maxlen - len(text_x), 0)
		for t, char in enumerate(text_x):
		    if t >= self.maxlen:
		        break
		    x[0, t + offset] = self.char_indices[char]
		pred = self.model.predict(x)
		return pred[0]