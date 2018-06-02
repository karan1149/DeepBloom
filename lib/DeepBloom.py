from BloomFilter import BloomFilter

class DeepBloom(object):
    def __init__(self, models, data):
        self.models = models
        self.train()
        self.bloomFilter = None

    def check(self, item):
        for model in models:
            if model.predict(item):
                return true
        return self.bloomFilter.check(item)

    def train(self):
        print("Training!")


## First model trained on all of the data
## Second model trained on false negatives from first model
