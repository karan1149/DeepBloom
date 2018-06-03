from DeepBloom import DeepBloom
from Model import Model
from AlmostPerfectModel import AlmostPerfectModel
from PerfectModel import PerfectModel
from GRUModel import GRUModel
import json
import random
import string
from utils import * 

## Get data to train on
with open('../data/dataset.json', 'r') as f:
    dataset = json.load(f)

positives = dataset['positives']
negatives = dataset['negatives']

def test_almost_perfect_model():
    fp_rate = .05

    train_dev_negatives = negatives[:int(.90 * len(negatives))]
    test_negatives = negatives[int(.90 * len(negatives)):]
    print("Number train, dev", len(train_dev_negatives))
    print("Number test", len(test_negatives))
    data = Data(positives, train_dev_negatives)

    # this parameter is not related to fp_rate
    db = DeepBloom(AlmostPerfectModel(.2), data, fp_rate)
    
    for positive in data.positives:
        assert(db.check(positive))

    false_positives = 0.0
    for negative in data.negatives:
        if db.check(negative):
            false_positives += 1
    print("Train/dev false Positive Rate: " + str(100* false_positives / len(train_dev_negatives)) + "%")

    false_positives = 0.0
    for neg in test_negatives:
        if db.check(neg):
            false_positives += 1
    print("Test false positive rate: ", str(100* false_positives / len(test_negatives)) + "%")

def test_gru_model(positives, negatives):
    fp_rate = 0.05

    train_dev_negatives = negatives[:int(.3 * len(negatives))]
    test_negatives = negatives[int(.3 * len(negatives)):int(.6 * len(negatives))]
    positives = positives[:int(.3 * len(positives))]
    print("Number train, dev", len(train_dev_negatives))
    print("Number test", len(test_negatives))
    print("Number positives ", len(positives))

    data = Data(positives, train_dev_negatives)

    db = DeepBloom(GRUModel('../data/glove.6B.50d-char.txt', 50, 0.001, pca_embedding_dim=16), data, fp_rate)

    for positive in positives:
        assert(db.check(positive))

    false_positives = 0.0
    for negative in data.negatives:
        if db.check(negative):
            false_positives += 1
    print("Train/dev false Positive Rate: " + str(100* false_positives / len(train_dev_negatives)) + "%")

    false_positives = 0.0
    for neg in test_negatives:
        if db.check(neg):
            false_positives += 1
    print("Test false positive rate: ", str(100* false_positives / len(test_negatives)) + "%")


# test_almost_perfect_model()
test_gru_model(positives, negatives)
