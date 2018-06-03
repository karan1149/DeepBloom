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
        count = 0.0
        false_positives = 0.0
        for negative in data.negatives:
            count += 1
            if db.check(negative):
                false_positives+= 1

        print("False Positive Rate: " + str(false_positives / count))

TestDeepBloom().test_almost_perfect_model()
