# Tobby Lie
# CSCI-5931 PA1
# August 31, 2019
# Last modified: 9/6/19 @ 12:25AM

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import backend as K
from keras import optimizers
from sklearn import preprocessing
from sklearn.metrics import f1_score
import numpy as np
import csv

# Metric precision, recall, f1 have been removed from 
# Keras 2.0 version so will need to create them for use
# functions below used from: 
# https://github.com/GeekLiB/keras/blob/master/keras/metrics.py
def precision_m(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall_m(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# NOTE: WHEN USING THIS FMEASURE FUNCTION IT WILL BE A LITTLE OFF BECAUSE
# K.EPSILON IS USED TO ENSURE NO DIVIDE BY ZERO ERROR OCCURS
def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
	precision = precision_m(y_true, y_pred)
	recall = recall_m(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))

def listOfLists(lst):
	''' Using a list comprehension create a list of lists based on lst input'''
	return [[el] for el in lst]

######################################################################################################################
# similar to above but used for dataset.csv

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

		# Need to separate data 80/20 split so some 
		# data is used to train and some to test 
		# this is to verify the effectiveness of our model
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
######################################################################################################################
# create list of lists from the labels
labels = listOfLists(labels)
test_labels = listOfLists(test_labels)
######################################################################################################################
# make each list into an np.array for ease of use with keras
training_examples = np.array(training_examples)
test_examples = np.array(test_examples)
labels = np.array(labels)
test_labels = np.array(test_labels)
######################################################################################################################
# feature scale to prepare data for training and testing
training_examples_scaled = preprocessing.scale(training_examples)
test_examples_scaled = preprocessing.scale(test_examples)
######################################################################################################################
# these lists will hold the metrics for each test for each model
accuracies = []
precisions = []
recalls = []
f1_scores = []
######################################################################################################################

# model 1
model_1 = Sequential()
model_1.add(Dense(4, activation='sigmoid', input_dim=8))
model_1.add(Dense(5, activation='sigmoid'))
model_1.add(Dense(4, activation='sigmoid'))
model_1.add(Dense(5, activation='sigmoid'))
model_1.add(Dense(4, activation='sigmoid'))
model_1.add(Dense(1, activation='sigmoid'))

model_1.compile(optimizer='Adam',
			loss='binary_crossentropy',
			metrics=['accuracy', fmeasure, precision_m, recall_m])

history_1 = model_1.fit(training_examples_scaled, labels, epochs=100, batch_size=32)
loss, accuracy, f1_score, precision, recall = model_1.evaluate(test_examples_scaled, test_labels, batch_size=32)

accuracies.append(str(accuracy))
precisions.append(str(precision))
recalls.append(str(recall))
f1_scores.append(str(f1_score))
######################################################################################################################
# model 2

model_2 = Sequential()
model_2.add(Dense(11, activation='relu', input_dim=8))
model_2.add(Dense(9, activation='relu'))
model_2.add(Dense(5, activation='relu'))
model_2.add(Dense(3, activation='sigmoid'))
model_2.add(Dense(1, activation='sigmoid'))

model_2.compile(optimizer='Adam',
			loss='binary_crossentropy',
			metrics=['accuracy', fmeasure, precision_m, recall_m])

history_2 = model_2.fit(training_examples_scaled, labels, epochs=100, batch_size=32)
loss, accuracy, f1_score, precision, recall = model_2.evaluate(test_examples_scaled, test_labels, batch_size=32)

accuracies.append(str(accuracy))
precisions.append(str(precision))
recalls.append(str(recall))
f1_scores.append(str(f1_score))
######################################################################################################################
# model 3

model_3 = Sequential()
model_3.add(Dense(11, activation='sigmoid', input_dim=8))
model_3.add(Dense(15, activation='sigmoid'))
model_3.add(Dense(5, activation='sigmoid'))
model_3.add(Dense(30, activation='sigmoid'))
model_3.add(Dense(1, activation='sigmoid'))

model_3.compile(optimizer='Adam',
			loss='binary_crossentropy',
			metrics=['accuracy', fmeasure, precision_m, recall_m])

history_3 = model_3.fit(training_examples_scaled, labels, epochs=1000, batch_size=32)
loss, accuracy, f1_score, precision, recall = model_3.evaluate(test_examples_scaled, test_labels, batch_size=32)

accuracies.append(str(accuracy))
precisions.append(str(precision))
recalls.append(str(recall))
f1_scores.append(str(f1_score))

######################################################################################################################
# model 4

model_4 = Sequential()
model_4.add(Dense(200, activation='sigmoid', input_dim=8))
model_4.add(Dense(150, activation='sigmoid'))
model_4.add(Dense(50, activation='relu'))
model_4.add(Dense(120, activation='sigmoid'))
model_4.add(Dense(1, activation='sigmoid'))

model_4.compile(optimizer='Adam',
			loss='binary_crossentropy',
			metrics=['accuracy', fmeasure, precision_m, recall_m])

history_4 = model_4.fit(training_examples_scaled, labels, epochs=100, batch_size=32)
loss, accuracy, f1_score, precision, recall = model_4.evaluate(test_examples_scaled, test_labels, batch_size=32)

accuracies.append(str(accuracy))
precisions.append(str(precision))
recalls.append(str(recall))
f1_scores.append(str(f1_score))
######################################################################################################################
# save model
model_4.save('model_4.h5')
model_json = model_4.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
######################################################################################################################
# display metrics for all of the tests
for index, val in enumerate(accuracies):
	print("------------------------------------------------------------------")
	print("Accuracy for model " + str(index + 1) + " test: " + str(accuracies[index]))
	print("Precision for model " + str(index + 1) + " test: " + str(precisions[index]))
	print("Recall for model " + str(index + 1) + " test: " + str(recalls[index]))
	print("F1 Score for model " + str(index + 1) + " test: " + str(f1_scores[index]))
	print("------------------------------------------------------------------")
######################################################################################################################