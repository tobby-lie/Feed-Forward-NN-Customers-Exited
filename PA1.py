# Tobby Lie
# CSCI-5931 PA1
# August 31, 2019
# Last modified: 9/3/19 @ 3:21PM

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import backend as K
from keras import optimizers
from sklearn import preprocessing
import numpy as np
import csv

# to print entire list with out dots
import sys
np.set_printoptions(threshold=sys.maxsize)
######################################################################################################################
# Metric precision, recall, f1 have been removed from 
# Keras 2.0 version so will need to create them for use
# functions below used from: 
# https://github.com/GeekLiB/keras/blob/master/keras/metrics.py
def precision_m(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall_m(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
        
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision_m(y_true, y_pred)
    r = recall_m(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    '''Calculates the f-measure, the harmonic mean of precision and recall.
    '''
    return fbeta_score(y_true, y_pred, beta=1)

def listOfLists(lst):
	return [[el] for el in lst]
######################################################################################################################
with open('judge.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')

	cust_ID = []

	credit_scores = []
	ages = []
	tenures = []
	balances = []
	num_products =[]
	has_cards = []
	active_members = []
	estimated_salaies = []

	features = [] 
	test_model_examples = []

	for row in readCSV:

		cust_ID.append(float(row[0]))

		credit_score = row[1]
		age = row[2]
		tenure = row[3]
		balance = row[4]
		num_product = row[5]
		has_card = row[6]
		active_member = row[7]
		estimated_salary = row[8]

		features.append(int(credit_score))
		features.append(int(age))
		features.append(int(tenure))
		features.append(float(balance))
		features.append(int(num_product))
		features.append(int(has_card))
		features.append(int(active_member))
		features.append(float(estimated_salary))

		test_model_examples.append(features[:])

		features.clear()
######################################################################################################################
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
######################################################################################################################
labels = listOfLists(labels)
test_labels = listOfLists(test_labels)
######################################################################################################################
training_examples = np.array(training_examples)
test_examples = np.array(test_examples)
labels = np.array(labels)
test_labels = np.array(test_labels)

test_model_examples = np.array(test_model_examples)
######################################################################################################################
training_examples_scaled = preprocessing.scale(training_examples)
test_examples_scaled = preprocessing.scale(test_examples)
test_model_examples = preprocessing.scale(test_model_examples)
######################################################################################################################
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
			metrics=['accuracy', recall_m, precision_m, fmeasure])

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
			metrics=['accuracy', recall_m, precision_m, fmeasure])

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
			metrics=['accuracy', recall_m, precision_m, fmeasure])

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
			metrics=['accuracy', recall_m, precision_m, fmeasure])

history_4 = model_4.fit(training_examples_scaled, labels, epochs=100, batch_size=32)
loss, accuracy, f1_score, precision, recall = model_4.evaluate(test_examples_scaled, test_labels, batch_size=32)

accuracies.append(str(accuracy))
precisions.append(str(precision))
recalls.append(str(recall))
f1_scores.append(str(f1_score))
######################################################################################################################

predictions = model_4.predict(test_model_examples)

for index, prediction in enumerate(predictions):
	if prediction[0] < 0.5:
		prediction[0] = 0
	elif prediction[0] > 0.5:
		prediction[0] = 1

prediction_pairs = [[] for x in range(len(predictions))]

for index, prediction in enumerate(predictions):
	prediction_pairs[index].append(cust_ID[index])
	prediction_pairs[index].append(prediction[0])

prediction_pairs = np.array(prediction_pairs)

np.savetxt("judge-pred.csv", prediction_pairs, delimiter=",", fmt=("%i, %i"))
######################################################################################################################
for index, val in enumerate(accuracies):
	print("------------------------------------------------------------------")
	print("Accuracy for model " + str(index + 1) + " test: " + str(accuracies[index]))
	print("Precision for model " + str(index + 1) + " test: " + str(precisions[index]))
	print("Recall for model " + str(index + 1) + " test: " + str(recalls[index]))
	print("F1 Score for model " + str(index + 1) + " test: " + str(f1_scores[index]))
	print("------------------------------------------------------------------")
######################################################################################################################







