from Model import Model
import random
import mmh3

DISCRETE_SIZE = 1000

# Model for testing purposes that stores a set of the values
# and has a given false positive rate by randomly predicting
# some negative values as positive
class AlmostPerfectModel(Model):
	def __init__(self, fp_rate, seed=None):
		self.table = set()
		self.fp_rate = float(fp_rate)
		if not seed:
			self.seed = random.randrange(0, 100)
		else:
			self.seed = seed

	def fit(self, X, y):
		for i, x in enumerate(X):
			if y[i] == 1:
				self.table.add(x)

	def predict(self, x):
		if x in self.table:
			return ((mmh3.hash(x, self.seed) % (DISCRETE_SIZE / 2)) + DISCRETE_SIZE / 2) / float(DISCRETE_SIZE)
		else:
			if mmh3.hash(x, self.seed) % DISCRETE_SIZE < self.fp_rate * DISCRETE_SIZE:
				return ((mmh3.hash(x, self.seed) % (DISCRETE_SIZE / 2)) + DISCRETE_SIZE / 2) / float(DISCRETE_SIZE)
			else:
				return (mmh3.hash(x, self.seed) % (DISCRETE_SIZE / 2)) / float(DISCRETE_SIZE)

	def predicts(self, X):
		return [self.predict(x) for x in X]
