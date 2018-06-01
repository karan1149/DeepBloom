## Adapted from https://www.geeksforgeeks.org/bloom-filters-introduction-and-python-implementation/

import math
import mmh3
from bitarray import bitarray

class BloomFilter(object):
    def __init__(self, items_count, fp_prob):
        self.fp_prob = fp_prob
        self.size = self.get_size(items_count, fp_prob)
        self.hash_count = self.get_hash_count(self.size, items_count)
        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)

    def add(self, item):
        for i in range(self.hash_count):
            digest = self.get_digest(item, i)
            self.bit_array[digest] = 1

    def check(self, item):
        for i in range(self.hash_count):
            digest = self.get_digest(item, i)
            if (self.bit_array[digest] == 0):
                return False
        return True

    def get_size(self,n,p):
        m = -(n * math.log(p))/(math.log(2)**2)
        return int(m)

    def get_hash_count(self, m, n):
        k = (m/n) * math.log(2)
        return int(k)

    def get_digest(self, item, index):
        return mmh3.hash(bytes(item), i) % self.size

bf = BloomFilter(100000, .5)
for i in range(100000):
    bf.add(i)
    if not bf.check(i):
        print("False Negative!")
