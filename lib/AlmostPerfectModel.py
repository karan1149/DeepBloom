from Model import Model
import random
import mmh3

DISCRETE_SIZE = 1000

# Model for testing purposes that stores a set of the values
# and has a given false positive rate by randomly predicting 
# some negative values as positive
class AlmostPerfectModel(Model):
	def __init__(self, fp_rate):
		self.table = set()
		self.fp_rate = float(fp_rate)

	def fit(self, X, y):
		for i, x in enumerate(X):
			if y[i] == 1:
				self.table.add(x)

	def predict(self, x):
		if x in self.table:
			return 1
		else:
			if mmh3.hash(x) % DISCRETE_SIZE < self.fp_rate * DISCRETE_SIZE:
				return 1
			else: 
				return 0