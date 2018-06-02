from Model import Model
import random

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
			if random.random() < self.fp_rate:
				return 1
			else: 
				return 0