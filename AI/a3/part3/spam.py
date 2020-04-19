#!/usr/local/bin/python3
# CS B551 Fall 2019, Assignment #3
#
# Your names and user ids:
#
# Kelly Wheeler (kellwhee), Neha Supe (nehasupe)
# 
#


import pandas as pd
import os
import re
import numpy as np
import sys

class NaiveBayes:
	classes = ['spam', 'notspam']
	tokens_dict = [0] * len(classes)
	all_tokens = {}
	prior_prob = [0] * len(classes)
	total_count = [0] * len(classes)
	predictions = []
	denominator = [0] * len(classes)

	# The likelihood probabities are calculated using laplace smoothing 
	# Referred blog:
	# https://medium.com/@theflyingmantis/text-classification-in-nlp-naive-bayes-a606bf419f8c

	def fit(self, X_train, y_train):
		# defined as an list of 2 dictinaries for tokens for both classes 
		for i in range(len(self.classes)):
			# calculating prior probabilities for each class
			class_samples = X_train[y_train == self.classes[i]]
			self.prior_prob[i] = np.sum(y_train == self.classes[i])/y_train.shape[0]
			# adding all words in the class to all_tokens dictionary which will have all tokens of the dataset
			for sample in class_samples:
				for token in sample.split():
					self.all_tokens[token] = 0

		# For each class storing the count of occurence of that word in that class 	
		for i in range(len(self.classes)):
			self.tokens_dict[i] = self.all_tokens
			for sample in class_samples:
				for token in sample.split():
					self.tokens_dict[i][token] += 1
			self.tokens_dict[i] = {key: value for key, value in self.tokens_dict[i].items() if value != 0}
		# laplace smoothing for the denomination part of the likelihood probability
		for i in range(len(self.classes)):
			self.denominator[i] = sum(list(self.tokens_dict[i].values())) + len(list(self.all_tokens.keys())) + 1

	def predict(self, y_test):
		for y in y_test:
			pred = 0
			posterior_prob = [0] * len(self.classes)
			likelihood_prob = [0] * len(self.classes)
			for i in range(len(self.classes)):
				tokens = y.split()
				for word in tokens:
					t = self.tokens_dict[i].get(word)
					# t will be none if the word has never occured in the dataset for that class
					# So we set t to 0 and later 1 is added to t which takes care of zero probabilities
					# This step is part of laplace smoothing
					if t == None:
						t = 0
					token_count = t + 1
					token_prob = token_count/ self.denominator[i]
					likelihood_prob[i] += np.log(token_prob)
				posterior_prob[i] = likelihood_prob[i] + np.log(self.prior_prob[i])
			# 
			if posterior_prob[0] > posterior_prob[1]:
				self.predictions.append('spam')
			else:
				self.predictions.append('notspam')
		return self.predictions


# text preprocesing
def data_preprocess(text):
	#text = text.strip()
	#text = text.replace(r'\n', " ")
	#text = text.lower()
	# text = re.sub('[\\W_]+', ' ', text)
	#text = re.sub('(\\s+)', ' ', text)	
	return text

files = []
data = []
testdat = []

# Referred for importing files in the program
# https://kite.com/python/examples/4293/os-get-the-relative-paths-of-all-files-and-subdirectories-in-a-directory

# Importing files from the training directory
# Importing files from notspam directory
for root, dirs, files in os.walk(sys.argv[1]+"/notspam"):
    for f in files:
    	F = open(sys.argv[1]+"/notspam/"+f, 'r', encoding="Latin-1")
    	data.append([F.read(), 'notspam'])
# Importing files from spam directory
for root, dirs, files in os.walk(sys.argv[1]+"/spam"):
    for f in files:
    	F = open(sys.argv[1]+"/spam/"+f, 'r', encoding="Latin-1")
    	data.append([F.read(), 'spam'])

# Putting all the training data into dataframe
train = pd.DataFrame(data, columns = ['email', 'class'])
print(train.shape)

for root, dirs, files in os.walk(sys.argv[2]):
    for f in files:
    	F = open(sys.argv[2]+"/"+f, 'r', encoding="Latin-1")
    	testdat.append([F.read()])
test = pd.DataFrame(testdat, columns = ['email'])
filenames = files

# Calculating importing the groundthruths file which is required to calculate accuracy 
F = open("test-groundtruth.txt", 'r')
content = F.readlines()
data1 = [x.strip().split(' ')[0] for x in content]
data2 = [x.strip().split(' ')[1] for x in content]
test['filename'] = data1
test['class'] = data2

truth = {}
for i in range(len(data1)):
	truth[data1[i]] = data2[i]



# train['email'] = train['email'].apply(lambda x: data_preprocess(x))
# test['email'] = test['email'].apply(lambda x: data_preprocess(x))

X_train = train['email']
y_train = train['class']
X_test = test['email']
y_test = test['class']

# training and predict Naive Bayes
nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

pred = {}
for i in range(len(filenames)):
	pred[filenames[i]] = predictions[i]


# Storing results in the output file
results = [0] * len(data1)
for i in range(len(filenames)):
	results[i] = (filenames[i], predictions[i])
f = open(sys.argv[3], "w")
for r in results:
	f.write(' '.join(str(t) for t in r) + '\n')
f.close()
accuracy = 0
for i in range(len(filenames)):
	p = pred[filenames[i]]
	t = truth[filenames[i]]
	if p == t:
		accuracy += 1

print("accuracy", accuracy/len(filenames)* 100)
