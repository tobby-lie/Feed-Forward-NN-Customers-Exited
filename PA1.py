# Tobby Lie
# CSCI-5931 PA1
# August 31, 2019
# Last modified: 9/2/19 @ 12:31PM

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import backend as K
from keras import optimizers
from sklearn import preprocessing
import numpy as np
import csv
import time

# to print entire list with out dots
import sys
np.set_printoptions(threshold=sys.maxsize)

# Metric precision, recall, f1 have been removed from 
# Keras 2.0 version so will need to create them for use
# functions below used from: 
# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
	"""recall metric"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def precision_m(y_true, y_pred):
	"""precision metric"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def f1_m(y_true, y_pred):
	"""f1 metric"""
	precision = precision_m(y_true, y_pred)
	recall = recall_m(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))

def listOfLists(lst):
	return [[el] for el in lst]

with open('dataset.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')

	credit_scores = []
	ages = []
	tenures = []
	balances = []
	num_products =[]
	has_cards = []
	active_members = []
	estimated_salaies = []

	labels = []
	test_labels = []

	features = []
	training_examples = []
	test_examples = []
	counter = 0
	for row in readCSV:

		label = row[8]
		credit_score = row[0]
		age = row[1]
		tenure = row[2]
		balance = row[3]
		num_product = row[4]
		has_card = row[5]
		active_member = row[6]
		estimated_salary = row[7]

		if counter < 7200:

			features.append(int(credit_score))
			features.append(int(age))
			features.append(int(tenure))
			features.append(float(balance))
			features.append(int(num_product))
			features.append(int(has_card))
			features.append(int(active_member))
			features.append(float(estimated_salary))

			training_examples.append(features[:])

			features.clear()

			labels.append(int(label))
		else:

			features.append(int(credit_score))
			features.append(int(age))
			features.append(int(tenure))
			features.append(float(balance))
			features.append(int(num_product))
			features.append(int(has_card))
			features.append(int(active_member))
			features.append(float(estimated_salary))

			test_examples.append(features[:])

			features.clear()

			test_labels.append(int(label))

		counter += 1

labels = listOfLists(labels)
test_labels = listOfLists(test_labels)

training_examples = np.array(training_examples)
test_examples = np.array(test_examples)
labels = np.array(labels)
test_labels = np.array(test_labels)

training_examples_scaled = preprocessing.scale(training_examples)
test_examples_scaled = preprocessing.scale(test_examples)

model_1 = Sequential()
model_1.add(Dense(4, activation='sigmoid', input_dim=8))
model_1.add(Dense(5, activation='sigmoid'))
model_1.add(Dense(4, activation='sigmoid'))
model_1.add(Dense(5, activation='sigmoid'))
model_1.add(Dense(4, activation='sigmoid'))
model_1.add(Dense(1, activation='sigmoid'))

adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model_1.compile(optimizer='Adam',
			loss='binary_crossentropy',
			metrics=['accuracy', recall_m, precision_m, f1_m])

history = model_1.fit(training_examples_scaled, labels, epochs=100, batch_size=32)
loss, accuracy, f1_score, precision, recall = model_1.evaluate(test_examples_scaled, test_labels, batch_size=32)

print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("f1_score: " + str(f1_score))


# model 2
'''
model_2 = Sequential()
model_2.add(Dense(11, activation='relu', input_dim=8))
model_2.add(Dense(9, activation='relu'))
model_2.add(Dense(5, activation='relu'))
model_2.add(Dense(3, activation='sigmoid'))
model_2.add(Dense(1, activation='sigmoid'))
model_2.compile(optimizer='rmsprop',
			loss='binary_crossentropy',
			metrics=['acc', precision_m, recall_m, f1_m])
model_2.fit(training_examples, labels, epochs=100000, batch_size=30)


'''




