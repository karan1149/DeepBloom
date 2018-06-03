import random
import mmh3

def shuffle_for_training(negatives, positives):
    a = [(i, 0) for i in negatives]
    b = [(i, 1) for i in positives]
    combined = a + b
    random.shuffle(combined)
    return list(zip(*combined))

def string_digest(item, index):
    return mmh3.hash(bytes(item, 'utf-8'), index)
