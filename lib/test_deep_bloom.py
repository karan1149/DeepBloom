from DeepBloom import DeepBloom
from Model import Model
from AlmostPerfectModel import AlmostPerfectModel
from PerfectModel import PerfectModel
import json
import random
import string

class Data(object):
    def __init__(self, positives, negatives):
        self.positives = positives
        self.negatives = negatives

class TestDeepBloom(object):
    def test_almost_perfect_model(self):
        fp_rate = .05

        ## Get data to train on
        with open('../data/dataset.json', 'r') as f:
        	dataset = json.load(f)

        positives = dataset['positives']
        negatives = dataset['negatives']

        train_dev_negatives = negatives[:int(.90 * len(negatives))]
        test_negatives = negatives[int(.90 * len(negatives)):]
        print("Number train, dev", len(train_dev_negatives))
        print("Number test", len(test_negatives))
        data = Data(positives, train_dev_negatives)

        # this parameter is not related to fp_rate
        db = DeepBloom(AlmostPerfectModel(.2), data, fp_rate)
        false_positives = 0.0

        for positive in data.positives:
            assert(db.check(positive))

        for negative in data.negatives:
            if db.check(negative):
                false_positives += 1

        print("Train/dev false Positive Rate: " + str(100* false_positives / len(train_dev_negatives)) + "%")

        false_positives = 0.0
        for neg in test_negatives:
            if db.check(neg):
                false_positives += 1

        
        print("Test false positive rate: ", str(100* false_positives / len(test_negatives)) + "%")

TestDeepBloom().test_almost_perfect_model()
