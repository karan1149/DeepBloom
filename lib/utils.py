import random
import mmh3
import numpy as np
from collections import Counter

class Data(object):
    def __init__(self, positives, negatives):
        self.positives = positives
        self.negatives = negatives

def shuffle_for_training(negatives, positives):
    a = [(i, 0) for i in negatives]
    b = [(i, 1) for i in positives]
    combined = a + b
    random.shuffle(combined)
    return list(zip(*combined))

def string_digest(item, index):
    return mmh3.hash(bytes(item, 'utf-8'), index)


def split_negatives(data, train_portion=0.9):
    size = len(data.negatives)
    s1 = data.negatives[0:int(train_portion * size)]
    s2 = data.negatives[int(train_portion * size):]
    return (s1, s2)

def vectorize_dataset(text_X, text_y, maxlen):
    # Adapted from Keras examples
    print("Vectorizing data...")
    raw_text = ''.join(text_X)
    print("Corpus length", len(raw_text))
    chars = sorted(list(set(raw_text)))
    print(chars)
    print('Total chars:', len(chars))

    lengths = [len(url) for url in text_X]
    counter = Counter(lengths)
    counts = sorted([(key, counter[key]) for key in counter])
    print(counts)
    max_seen = 0
    for url in text_X:
        max_seen = max(len(url), max_seen)
    print("max seen length of URL", max_seen) 
    print("Using maxlen", maxlen)
    char_indices = dict((c, i + 1) for i, c in enumerate(chars))
    indices_char = dict((i + 1, c) for i, c in enumerate(chars))

    # 0 in this indicates empty word, 1 through len(chars) inclusive
    # indicates a particular char
    X = np.zeros((len(text_X), maxlen), dtype=np.int)
    y = np.zeros((len(text_X)), dtype=np.bool)
    for i, url in enumerate(text_X):
        offset = max(maxlen - len(url), 0)
        for t, char in enumerate(url):
            if t >= maxlen:
                break
            X[i, t + offset] = char_indices[char]
        y[i] = 1 if text_y[i] == 1 else 0

    return X, y, char_indices, indices_char

def test_model(model, text_X, text_y):
    total = float(len(text_X))
    total_correct = 0.0
    false_positives = 0.0
    false_negatives = 0.0
    for i, url in enumerate(text_X):
        raw_pred = model.predict(url)
        pred = 1 if raw_pred > 0.5 else 0
        label = text_y[i]
        if pred == label:
            total_correct += 1
        else:
            if pred == 1:
                false_positives += 1
            else: 
                false_negatives += 1
    return total_correct / total, false_positives / total, false_negatives / total
