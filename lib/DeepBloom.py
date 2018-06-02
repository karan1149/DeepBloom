from BloomFilter import BloomFilter
import math
import random

class DeepBloom(object):
    def __init__(self, model, data, fp_rate):
        self.model = model
        self.threshold = 0
        self.fp_rate = fp_rate
        self.data = data
        self.train()
        self.bloomFilter = None

    def check(self, item):
        if self.model.predict(item) > self.threshold:
            return true
        return self.bloomFilter.check(item)


    ## For now, only train the first model.
    def train(self):

        ## Split negative data into subgroups.
        (s1, s2, s3) = self.split_negatives()

        ## Shuffle together subset of negatives and positives.
        ## Then, train the model on this data.
        shuffled = self.shuffle_for_training(s1, self.data.positives)
        self.model.fit(shuffled[0], shuffled[1])

        ## We want a threshold such that s2.size/2 elements
        ## are greater than threshold.
        fp_index = int((len(s2) * self.fp_rate/2) + 1)
        predictions = [self.model.predict(item) for item in s2]
        list.sort(predictions)
        threshold = predictions[fp_index]
        print(threshold)


    def split_negatives(self):
        size = len(self.data.negatives)
        s1 = self.data.negatives[0:math.floor(.8*size)]
        s2 = self.data.negatives[math.ceil(.8*size):math.floor(.9*size)]
        s3 = self.data.negatives[math.ceil(.9*size):]
        return (s1, s2, s3)

    def shuffle_for_training(self, negatives, positives):
        a = [(i, 0) for i in negatives]
        b = [(i, 1) for i in positives]
        combined = a + b
        random.shuffle(combined)
        return combined
