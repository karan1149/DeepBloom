from BloomFilter import BloomFilter
import math
import random
from utils import *
import mmh3

class DeepBloom(object):
    def __init__(self, model, data, fp_rate):
        self.model = model
        self.threshold = 0
        self.fp_rate = float(fp_rate)
        self.fit(data)
        self.create_bloom_filter(data)

    def check(self, item):
        # print(self.model.predict(item))
        if self.model.predict(item) > self.threshold:
            return True
        return self.bloom_filter.check(item)

    def create_bloom_filter(self, data):
        self.bloom_filter = BloomFilter(
            len(data.negatives),
            2*self.fp_rate,
            string_digest
        )
        for positive in data.positives:
            if self.model.predict(positive) <= self.threshold:
                self.bloom_filter.add(positive)
                assert(self.bloom_filter.check(positive))


    ## For now, only train the first model.
    def fit(self, data):

        ## Split negative data into subgroups.
        (s1, s2) = self.split_negatives(data)

        ## Shuffle together subset of negatives and positives.
        ## Then, train the model on this data.
        shuffled = shuffle_for_training(s1, data.positives)
        self.model.fit(list(shuffled[0]), list(shuffled[1]))

        ## We want a threshold such that at most s2.size * fp_rate/2 elements
        ## are greater than threshold.
        fp_index = len(s2) - int(math.ceil((len(s2) * self.fp_rate/2)))
        predictions = [self.model.predict(item) for item in s2]
        list.sort(predictions)
        self.threshold = predictions[fp_index]
        print(self.threshold)


    def split_negatives(self, data):
        size = len(data.negatives)
        s1 = data.negatives[0:math.floor(.8*size)]
        s2 = data.negatives[math.floor(.8*size):math.floor(.9*size)]
        return (s1, s2)
