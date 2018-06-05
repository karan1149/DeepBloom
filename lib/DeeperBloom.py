from BloomFilter import BloomFilter
import math
import random
from utils import *
import mmh3

class DeeperBloom(object):
    '''
    fp_fractions is an array of size self.k + 1 containing the fraction of the
    fp_rate dedicated to each of k models, then bloom filter, in that order. If not passed,
    default to splitting false positives evenly.
    '''
    def __init__(self, models, data, fp_rate, fp_fractions=None):
        self.models = models
        self.k = len(self.models)
        self.thresholds = [None] * self.k
        if fp_fractions is None:
            self.fp_rate_bloom = float(fp_rate) / (self.k + 1)
            self.fp_rates = [float(fp_rate) / (self.k + 1)] * self.k
        else:
            self.fp_rate_bloom = fp_fractions[self.k] * fp_rate
            self.fp_rates = []
            for i in range(self.k):
                self.fp_rates.append(fp_fractions[i] * fp_rate)
        self.fit(data)
        self.create_bloom_filter(data)

    def check(self, item):
        for i in range(self.k):
            if self.models[i].predict(item) > self.thresholds[i]:
                return True
        return self.bloom_filter.check(item)

    def create_bloom_filter(self, data):
        print("Creating bloom filter")
        false_negatives = []
        preds = []
        for i in range(self.k):
            preds.append(self.models[i].predicts(data.positives))
        for j in range(len(data.positives)):
            is_false = True
            for i in range(self.k):
                pred = preds[i][j]
                if pred > self.thresholds[i]:
                    is_false = False
            if is_false:      
                false_negatives.append(data.positives[j])
        print("Number of false negatives at bloom time", len(false_negatives))
        print("Effective bloom filter false positive rate", self.fp_rate_bloom)
        self.bloom_filter = BloomFilter(
            len(false_negatives),
            self.fp_rate_bloom,
            string_digest
        )
        for fn in false_negatives:
            self.bloom_filter.add(fn)
        print("Created bloom filter")

    def fit(self, data):
        ## Split negative data into subgroups.

        for i in range(self.k):
            # First prep s1, s2 and curr_positives
            print("Data prep", i)
            if i == 0:
                (s1, s2) = split_negatives(data)
                curr_positives = data.positives
                false_negatives = curr_positives
            else:
                # TODO BALANCE
                # TODO FIX curr_positives not carrying through all false negatives
                # TODO add back difficulty factor stuff?
                # DIFFICULTY_FACTOR = 1.2
                # Get false negatives from curr_positives, with
                # respect to prev model
                new_false_negatives = []
                new_positives = []
                preds = self.models[i - 1].predicts(false_negatives)
                for j in range(len(false_negatives)):
                    pred = preds[j]
                    if pred <= self.thresholds[i - 1]:
                        new_false_negatives.append(false_negatives[j])
                        if pred <= self.thresholds[i - 1]:
                            new_positives.append(false_negatives[j])
                curr_positives = new_positives
                false_negatives = new_false_negatives

                # Get true negatives from s1, with respect to prev
                # model
                new_s1 = []
                preds = self.models[i - 1].predicts(s1)
                for j in range(len(s1)):
                    pred = preds[j]
                    if pred <= self.thresholds[i - 1]:
                        new_s1.append(s1[j])
                s1 = new_s1

                # Get true negatives from s2, with respect to prev
                # model
                new_s2 = []
                preds = self.models[i - 1].predicts(s2)
                for j in range(len(s2)):
                    pred = preds[j]
                    if pred <= self.thresholds[i - 1]:
                        new_s2.append(s2[j])
                s2 = new_s2

                # Ensure that s1 is balanced relative to curr_positives
                # if (len(s1) > len(curr_positives)):
                #     s1 = s1[:len(curr_positives)]
            print("Number of false negatives at this step", len(false_negatives))
            print("Training model with train, dev, positives", i, len(s1), len(s2), len(curr_positives))
            random.shuffle(s1)
            random.shuffle(s2)
            random.shuffle(curr_positives)

            ## Shuffle together subset of negatives and positives.
            ## Then, train the model on this data.
            shuffled = shuffle_for_training(s1, curr_positives)
            self.models[i].fit(shuffled[0], shuffled[1])
            print("Done fitting")

            ## We want a threshold such that at most s2.size * fp_rates[i] elements
            ## are greater than threshold.
            print("Using effective false positive rate for model", i, self.fp_rates[i])
            fp_index = math.ceil((len(s2) * (1 - self.fp_rates[i])))
            predictions = self.models[i].predicts(s2)
            predictions.sort()
            self.thresholds[i] = predictions[fp_index]
            print("Threshold for model", i, self.thresholds[i])

