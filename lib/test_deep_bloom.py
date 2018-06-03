from DeepBloom import DeepBloom
from Model import Model
from AlmostPerfectModel import AlmostPerfectModel
import random
import string

class Data(object):
    def __init__(self):
        self.negatives = [self.generate_random_string(20) for i in range(10000)]
        self.positives = [self.generate_random_string(20) for i in range(10001,20001)]

    def generate_random_string(self, N):
        return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

class TestDeepBloom(object):
    def test_almost_perfect_model(self):
        fp_rate = .05
        data = Data()
        db = DeepBloom(AlmostPerfectModel(fp_rate), data, fp_rate)
        false_positives = 0.0
        false_negatives = 0.0

        for positive in data.positives:
            if not db.check(positive):
                false_negatives+= 1

        for negative in data.negatives:
            if db.check(negative):
                false_positives+= 1

        print("False Negative Rate: " + str(100* false_negatives / len(data.positives)) + "%")
        print("False Positive Rate: " + str(100* false_positives / len(data.negatives)) + "%")

TestDeepBloom().test_almost_perfect_model()
