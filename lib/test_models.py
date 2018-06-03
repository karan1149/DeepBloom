from PerfectModel import PerfectModel
from AlmostPerfectModel import AlmostPerfectModel
from GRUModel import GRUModel
import json
from utils import *
from keras.models import load_model
import math

def test_perfect_model(positives, negatives):
    print(len(positives))

    negatives_train = negatives[0: int(len(negatives) * .8)]
    negatives_test = negatives[int(len(negatives) * .8): ]

    print(len(negatives_train), len(negatives_test))

    model = PerfectModel()
    # should shuffle but doesn't matter in this case
    model.fit(positives + negatives_train, [1 for _ in range(len(positives))] + [0 for _ in range(len(negatives_train))])

    for x in positives:
        assert(model.predict(x) == 1)

    for x in negatives_test + negatives_train:
        if model.predict(x) != 0:
            print(x)
            print(x in positives)
            print(x in negatives_train)
            print(x in negatives_test)
        assert(model.predict(x) == 0)


def test_almost_perfect_model(positives, negatives):
    print(len(positives))

    negatives_train = negatives[0: int(len(negatives) * .8)]
    negatives_test = negatives[int(len(negatives) * .8): ]

    print(len(negatives_train), len(negatives_test))

    model = AlmostPerfectModel(0.05)
    print("Using almost perfect false positive rate of 0.05")
    # should shuffle but doesn't matter in this case
    model.fit(positives + negatives_train, [1 for _ in range(len(positives))] + [0 for _ in range(len(negatives_train))])

    for x in positives:
        assert(model.predict(x) >= 0.5)

    false_positives_train = 0.0
    for x in negatives_train:
        if model.predict(x) >= 0.5:
         false_positives_train += 1

    false_positives_test = 0.0
    for x in negatives_test:
        if model.predict(x) >= 0.5:
         false_positives_test += 1

    print(false_positives_train / len(negatives_train), "false positive rate for train.")
    print(false_positives_test / len(negatives_test), "false positive rate for test.") 


def test_gru_model(positives, negatives, data_fraction=1.0, fp_rate=0.01, lr=0.001, pca_embedding_dim=None, maxlen=50, gru_size=16, batch_size=1024, hidden_size=None, second_gru_size=None):
    positives = positives[:int(data_fraction * len(positives))]
    negatives = negatives[:int(data_fraction * len(negatives))]

    negatives_train = negatives[0: int(len(negatives) * .8)]
    negatives_dev = negatives[int(len(negatives) * .8): int(len(negatives) * .9)]
    negatives_test = negatives[int(len(negatives) * .9): ]

    print("Split sizes:")
    print(len(positives), len(negatives_train), len(negatives_dev), len(negatives_test))

    model = GRUModel('../data/glove.6B.50d-char.txt', 50, lr=lr, pca_embedding_dim=pca_embedding_dim, maxlen=maxlen, gru_size=gru_size, batch_size=batch_size, hidden_size=hidden_size, second_gru_size=second_gru_size)
    shuffled = shuffle_for_training(negatives_train, positives)

    model.fit(shuffled[0], shuffled[1])
    # print("Params", model.model.count_params())
    # model.save('model_test.h5')
    # model = load_model('model_test.h5')

    print("Using threshold 0.5")

    threshold = 0.5

    evaluate_model(model, positives, negatives_train, negatives_dev, negatives_test, threshold)

    print("Getting threshold for fp_rate", fp_rate)

    preds = model.predicts(negatives_dev)
    preds.sort()
    fp_index = math.ceil((len(negatives_dev) * (1 - fp_rate)))
    threshold = preds[fp_index]

    print("Using threshold", threshold) 

    evaluate_model(model, positives, negatives_train, negatives_dev, negatives_test, threshold)

if __name__=='__main__':

    with open('../data/dataset.json', 'r') as f:
        dataset = json.load(f)

    positives = dataset['positives']
    negatives = dataset['negatives']

    print("Testing perfect...")
    test_perfect_model(positives, negatives)
    print("Testing almost perfect...")
    test_almost_perfect_model(positives, negatives)
    print("Testing GRU model...")
    test_gru_model(positives, negatives, data_fraction=0.5, fp_rate=0.01, pca_embedding_dim=16, lr=0.001, maxlen=50, gru_size=16)


