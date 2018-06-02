from PerfectModel import PerfectModel
import json

with open('../data/dataset.json', 'r') as f:
	dataset = json.load(f)

positives = dataset['positives']
negatives = dataset['negatives']

def test_perfect_model():
	print(len(positives))

	negatives_train = negatives[0: int(len(negatives) * .8)]
	negatives_test = negatives[int(len(negatives) * .8): ]

	print(len(negatives_train), len(negatives_test))

	model = PerfectModel()

	model.fit(positives + negatives_train, [1 for _ in range(len(positives))] + [0 for _ in range(len(negatives_train))])

	for x in positives:
		assert(model.predict(x) == 1)

	for x in negatives_test + negatives_train:
		if model.predict(x) != 0:
			print(x)
			print(x in positives)
			print(x in negatives_train)
			print(x in negatives_test)
		assert(model.predict(x) == 0)

if __name__=='__main__':
	test_perfect_model()


