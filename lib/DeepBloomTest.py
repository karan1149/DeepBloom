from DeepBloom import DeepBloom
from Model import Model

class Data(object):
    def __init__(self):
        self.negatives = [i for i in range(100000)]
        self.positives = [i for i in range(100001,200001)]
db = DeepBloom(Model(), Data(), .05)
