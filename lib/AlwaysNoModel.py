from Model import Model

class AlwaysNoModel(Model):
	def __init__(self):
		self.table = set()

	def fit(self, X, y):
		pass

	def predict(self, x):
		return 0