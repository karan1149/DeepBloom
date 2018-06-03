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
        fp_rate = .20

        ## Get data to train on
        with open('../data/dataset.json', 'r') as f:
        	dataset = json.load(f)

        positives = dataset['positives']
        negatives = dataset['negatives']
        data = Data(positives, negatives)

        db = DeepBloom(AlmostPerfectModel(fp_rate), data, fp_rate)
        false_positives = 0.0
        false_negatives = 0.0

        for positive in data.positives:
            assert(db.check(positive))

        for negative in data.negatives:
            if db.check(negative):
                false_positives+= 1

        print("False Negative Rate: " + str(100* false_negatives / len(data.positives)) + "%")
        print("False Positive Rate: " + str(100* false_positives / len(data.negatives)) + "%")

TestDeepBloom().test_almost_perfect_model()
