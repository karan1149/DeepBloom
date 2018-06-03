from BloomFilter import BloomFilter
import math
import random
from utils import *
import mmh3

class DeeperBloom(object):
    def __init__(self, models, data, fp_rate):
        self.models = models
        self.k = len(self.models)
        self.thresholds = [None] * self.k
        self.fp_rate = float(fp_rate)
        self.fit(data)
        self.create_bloom_filter(data)

    def check(self, item):
        for i in range(self.k):
            if self.models[i].predict(item) > self.thresholds[i]:
                return True
        return self.bloom_filter.check(item)

    def create_bloom_filter(self, data):
        false_negatives = []
        for positive in data.positives:
            is_false = True
            for i in range(self.k):
                if self.models[i].predict(positive) > self.thresholds[i]:
                    is_false = False
            if is_false:      
                false_negatives.append(positive)
        print("Number of false negatives at bloom time", len(false_negatives))
        self.bloom_filter = BloomFilter(
            len(false_negatives),
            self.fp_rate / (self.k + 1),
            string_digest
        )
        for fn in false_negatives:
            self.bloom_filter.add(fn)

    def fit(self, data):
        ## Split negative data into subgroups.

        for i in range(self.k):
            # First prep s1, s2 and curr_positives
            if i == 0:
                (s1, s2) = split_negatives(data)
                curr_positives = data.positives
            else:
                # Get false negatives from curr_positives, with
                # respect to prev model
                false_negatives = []
                for pos in curr_positives:
                    pred = self.models[i - 1].predict(pos)
                    if pred <= self.thresholds[i - 1]:
                        false_negatives.append(pos)
                curr_positives = false_negatives

                # Get true negatives from s1, with respect to prev
                # model
                new_s1 = []
                for neg in s1:
                    pred = self.models[i - 1].predict(neg)
                    if pred <= self.thresholds[i - 1]:
                        new_s1.append(neg)
                s1 = new_s1

                # Get true negatives from s2, with respect to prev
                # model
                new_s2 = []
                for neg in s2:
                    pred = self.models[i - 1].predict(neg)
                    if pred <= self.thresholds[i - 1]:
                        new_s2.append(neg)
                s2 = new_s2

            print("Training model with train, dev, positives", i, len(s1), len(s2), len(curr_positives))

            ## Shuffle together subset of negatives and positives.
            ## Then, train the model on this data.
            shuffled = shuffle_for_training(s1, curr_positives)
            self.models[i].fit(shuffled[0], shuffled[1])

            print("s1 results", test_model(self.models[i], s1, [0 for _ in range(len(s1))]))
            print("s2 results", test_model(self.models[i], s2, [0 for _ in range(len(s2))]))
            print("pos results", test_model(self.models[i], curr_positives, [1 for _ in range(len(curr_positives))]))

            ## We want a threshold such that at most s2.size * fp_rate/2 elements
            ## are greater than threshold.
            fp_index = math.ceil((len(s2) * (1 - self.fp_rate/(self.k + 1))))
            predictions = [self.models[i].predict(item) for item in s2]
            predictions.sort()
            self.thresholds[i] = predictions[fp_index]

