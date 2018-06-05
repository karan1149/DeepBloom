import json
import os
import random
random.seed(42)
import argparse
import string

parser = argparse.ArgumentParser()
parser.add_argument("--augment", action="store_true", default=False, help="Whether to augment dataset by adding www. and removing www. when possible.")
args = parser.parse_args()

dataset_path = 'shallalist/'
desired_positive_categories = ['porn', 'models', 'sex/education', 'sex/lingerie']

folders = sorted([tup for tup in os.walk(dataset_path)])

nonallowed_characters = ['_', '&', '#', ';', '/', 'ü', ',', 'ö', '"', 'ı']

desired_negative_categories = []

for tup in folders:
	if 'domains' in tup[2] and 'urls' in tup[2]:
		desired_negative_categories.append(tup[0][len(dataset_path):])

desired_negative_categories = list(set(desired_negative_categories) - set(desired_positive_categories))

print(desired_positive_categories, desired_negative_categories)

# Generates a JSON dataset for positives and negatives from the Shallist set of blacklists
def generate_dataset(dataset_path, desired_positive_categories, desired_negative_categories, save_path="dataset.json"):

	# Validate category names and file structure
	for cat in desired_positive_categories + desired_negative_categories:
		assert(os.path.isfile(dataset_path + cat + "/domains"))

	assert(len(set(desired_positive_categories + desired_negative_categories)) == len(desired_positive_categories) + len(desired_negative_categories))

	positives = []
	negatives = []
	for cat in desired_positive_categories + desired_negative_categories:
		with open(dataset_path + cat + "/domains", 'r') as f:
			raw_urls = f.read()
		urls = raw_urls.split()
		if cat in desired_positive_categories:
			positives += urls
		else:
			negatives += urls

	# Remove unallowed characters

	initial_positive_length = len(positives)
	initial_negative_length = len(negatives)

	new_positives = []
	for url in positives:
		allowed = True
		for char in nonallowed_characters:
			if char in url:
				allowed = False
		if allowed:
			new_positives.append(url)
	positives = new_positives

	new_negatives = []
	for url in negatives:
		allowed = True
		for char in nonallowed_characters:
			if char in url:
				allowed = False
		if allowed:
			new_negatives.append(url)
	negatives = new_negatives

	print(initial_positive_length - len(positives), "invald character positives removed.")
	print(initial_negative_length - len(negatives), "invalid character negatives removed.")



	# Remove duplicates

	initial_positive_length = len(positives)
	initial_negative_length = len(negatives)

	positives = [pos.lower() for pos in positives]
	negatives = [neg.lower() for neg in negatives]

	# Ensures no duplicates within lists
	positives = set(positives)
	negatives = set(negatives)

	print(initial_positive_length - len(positives), "duplicate positives removed.")
	print(initial_negative_length - len(negatives), "duplicate negatives removed.")


	# augment if needed
	if args.augment:
		print("Before augmentation", len(positives), len(negatives))
		new_positives = set()
		for pos in positives:
			new_positives.add(pos)
			if random.random() < .9 and should_add_www(pos):
				new_positives.add('www.' + pos)
		positives = new_positives

		new_negatives = set()
		for neg in negatives:
			new_negatives.add(neg)
			if random.random() < .8 and should_add_www(neg):
				new_negatives.add('www.' + neg)
			if random.random() < .2:
				generated = generate_random_url(random.choice([i for i in range(8, 30)]))
				new_negatives.add(generated)
		negatives = new_negatives

	count = 0
	for x in positives:
		if x in negatives:
			negatives.remove(x)
			count += 1
	print(count, "duplicates removed across lists by removing from negatives.")


	positives = list(positives)
	negatives = list(negatives)

	random.shuffle(positives)
	random.shuffle(negatives)

	print("Number of positives:", len(positives))
	print("Number of negatives:", len(negatives))

	# Ensures no duplicates across lists
	assert(len(set(positives + negatives)) == len(positives) + len(negatives))

	with open(save_path, 'w') as f:
		json.dump({"positives": positives, "negatives": negatives}, f)

	print("Finished!")

def should_add_www(url):
	if url.startswith('www'):
		return False
	first_letter = ord(url[0])
	if first_letter < 97 or first_letter > 122:
		return False
	return True

def generate_random_url(N):
	if random.random() < 0.95:
		generated = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N)) + '.com'
	else:
		generated = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N)) + '.net'
	return generated

if __name__=='__main__':
	generate_dataset(dataset_path, desired_positive_categories, desired_negative_categories)