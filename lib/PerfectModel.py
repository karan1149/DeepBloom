from Model import Model

class PerfectModel(Model):
	def __init__(self):
		self.table = set()

	def fit(self, X, y):
		for i, x in enumerate(X):
			if y[i] == 1:
				self.table.add(x)

	def predict(self, x):
		return 1 if x in self.table else 0