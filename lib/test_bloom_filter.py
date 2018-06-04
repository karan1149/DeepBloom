from BloomFilter import BloomFilter
import mmh3
import string
import random
import json

def get_digest(item, index):
    return mmh3.hash(bytes(item), index)

def string_digest(item, index):
    return mmh3.hash(bytes(item, 'utf-8'), index)

def generate_random_string(N):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

#### Tests
def simple_test():
    bf = BloomFilter(1000, .001, get_digest)
    for i in range(1000):
        bf.add(i)
        if not bf.check(i):
            print("False Negative!")

    count = 0.0
    fp = 0.0
    for i in range(1001, 10000):
        if bf.check(i):
            fp+=1
        count += 1

    print("False Positive Rate: " + str(fp / count))

def string_test():
    bf = BloomFilter(1000, .001, string_digest)
    for i in range(1000):
        random_string = generate_random_string(i)
        bf.add(random_string)
        assert(bf.check(random_string))

    count = 0.0
    fp = 0.0
    for i in range(1001, 10000):
        random_string = generate_random_string(i)
        if bf.check(random_string):
            fp+=1
        count += 1

    print("False Positive Rate: " + str(fp / count))


def url_test(positives, negatives, fp_rate):
    bf = BloomFilter(len(positives), fp_rate, string_digest)
    for pos in positives:
        bf.add(pos)
        assert(bf.check(pos))
    print("Bits needed", bf.size)
    print("Hash functions needed", bf.hash_count)

    fp = 0.0
    for neg in negatives:
        if bf.check(neg):
            fp += 1
    print("False positives", fp / len(negatives))
    
if __name__=='__main__':
    with open('../data/dataset.json', 'r') as f:
        dataset = json.load(f)

    positives = dataset['positives']
    negatives = dataset['negatives']
    print(len(positives), len(negatives))
    url_test(positives, negatives, 0.001)
