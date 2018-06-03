from PerfectModel import PerfectModel
from AlmostPerfectModel import AlmostPerfectModel
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
	# should shuffle but doesn't matter in this case
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


def test_almost_perfect_model():
	print(len(positives))

	negatives_train = negatives[0: int(len(negatives) * .8)]
	negatives_test = negatives[int(len(negatives) * .8): ]

	print(len(negatives_train), len(negatives_test))

	model = AlmostPerfectModel(0.05)
	print("Using almost perfect false positive rate of 0.05")
	# should shuffle but doesn't matter in this case
	model.fit(positives + negatives_train, [1 for _ in range(len(positives))] + [0 for _ in range(len(negatives_train))])

	for x in positives:
		assert(model.predict(x) >= 0.5)

	false_positives_train = 0.0
	for x in negatives_train:
		if model.predict(x) >= 0.5:
		 false_positives_train += 1

	false_positives_test = 0.0
	for x in negatives_test:
		if model.predict(x) >= 0.5:
		 false_positives_test += 1

	print(false_positives_train / len(negatives_train), "false positive rate for train.")
	print(false_positives_test / len(negatives_test), "false positive rate for test.") 

if __name__=='__main__':
	print("Testing perfect...")
	test_perfect_model()
	print("Testing almost perfect...")
	test_almost_perfect_model()


