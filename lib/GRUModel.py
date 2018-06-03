from Model import Model
import utils

class GRUModel(Model):
	def __init__(self, embeddings_path, embedding_dim, lr):
		self.embeddings_path = embeddings_path
		self.embedding_dim = embedding_dim
		self.lr = lr

	def fit(self, X, y):
		X, y = vectorize_dataset(X, y)

	def predict(self, x):
		pass