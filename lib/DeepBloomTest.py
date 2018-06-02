from DeepBloom import DeepBloom
from Model import Model
import random
import string

class Data(object):
    def __init__(self):
        self.negatives = [self.generate_random_string(20) for i in range(10000)]
        self.positives = [self.generate_random_string(20) for i in range(10001,20001)]

    def generate_random_string(self, N):
        return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

db = DeepBloom(Model(), Data(), .05)
