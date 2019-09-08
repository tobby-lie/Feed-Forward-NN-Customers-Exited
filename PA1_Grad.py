# Tobby Lie
# CSCI-5931 PA1_Grad
# August 31, 2019
# Last modified: 9/4/19 @ 8:18PM

from sklearn import preprocessing
from random import random
from random import seed
import math
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

def sigmoid(x):
	return 1/(1+math.exp(-x))
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

learning_rate = 0.5
for index, label in enumerate(labels):
	input_layer = training_examples_scaled[index]
	weights_1 = np.random.rand(len(input_layer), 3)
	zVal_1 = np.dot(input_layer, weights_1)

	activation_1 = []
	for zVal in zVal_1:
		activation_1.append(sigmoid(zVal))

	weights_2 = np.random.rand(len(activation_1), 1)

	output_layer = sigmoid(np.dot(activation_1, weights_2))

	if label == 1:
		loss = -(math.log(output_layer))
	elif label == 0:
		loss = -(math.log(1-output_layer))

	the_label = float(label)
	error_last = (float(the_label) - float(output_layer))*(float(output_layer) 
		* (1.0 - float(output_layer)))

	error_hidden = np.zeros(shape=(1,3))
	for index, activation in enumerate(activation_1):
		error_hidden[0][index] = (activation*weights_2[index][0])

	# between input and hidden layer
	delta_1 = np.zeros((len(input_layer), 3))

	activation_1 = np.array(activation_1)
	delta_1 = delta_1+(activation_1.transpose()*error_hidden)

	delta_2 = np.zeros((len(activation_1), 1))
	output_layer = np.array(output_layer)
	delta_2 = delta_2+(output_layer.transpose()*error_last)

	input_layer_mult = []
	input_layer_mult.append(input_layer)
	input_layer_mult = np.array(input_layer_mult)

	temp = np.dot(error_hidden.transpose(), input_layer_mult)
	temp = np.dot(learning_rate, temp)
	weights_1 = np.add(temp.transpose(), weights_1)

####### trying to calculate weight = weight + learning_rate * error * input from
######https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
	print(activation_1.shape)

	temp = np.dot(error_last, activation_1)
	temp = np.dot(learning_rate, temp)
	weights_2 = np.add(temp, weights_2)

	print(weights_2)



######################################################################################################################
'''

current_training_example = 0
learning_rate = 0.5
true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0
m = len(labels)


	# Create random weights for each perceptron in hidden layer 1
	# Meaning 4 (perceptrons) x 8 (inputs) = 32 weights total
	# And 8 weights connecting to each perceptron in the hidden layer 1

for index, training_example_scaled in enumerate(training_examples_scaled):
	# list to hold the 4 perceptrons of hidden layer 1
	hidden_layer1_perceptrons = []
	# Run for loop 4 times, once for every perceptron in hidden layer 1
	for x in range(0, 4):
		# randomly create 8 weights for each perceptron connecting 
		# the given perceptron to each input with a weight
		random_weights = np.random.rand(8, 1)
		hidden_layer1_perceptrons.append(random_weights)

	hidden_layer1_perceptrons = np.array(hidden_layer1_perceptrons)

	# activation inputs for hidden layer 1
	activation_inputs_1 = []
	for indx, hidden_layer1_perceptron in enumerate(hidden_layer1_perceptrons):
		temp =[]
		temp.append(training_examples_scaled[current_training_example])
		temp = np.array(temp)
		activation_input = np.dot(temp,hidden_layer1_perceptron)
		activation_inputs_1.append(activation_input[0])

	activation_inputs_1 = np.array(activation_inputs_1)

	# Outputs for each perceptron in hidden layer 1
	# after sigmoid has been applied
	activation_outputs_1 = []
	for activation_input in activation_inputs_1:
		activation_output = sigmoid(activation_input)
		activation_output_temp = []
		activation_output_temp.append(activation_output)
		activation_outputs_1.append(activation_output_temp)

	activation_outputs_1 = np.array(activation_outputs_1)
	activation_outputs_1 = np.transpose(activation_outputs_1)

	######################################################################################################################

	# Hidden layer 2

	# list to hold the 4 perceptrons of hidden layer 1
	hidden_layer2_perceptrons = []
	# Run for loop 5 times, once for every perceptron in hidden layer 2
	for x in range(0, 5):
		# randomly create 4 weights for each perceptron connecting 
		# the given perceptron to each input with a weight
		random_weights = np.random.rand(4, 1)
		hidden_layer2_perceptrons.append(random_weights)

	hidden_layer2_perceptrons = np.array(hidden_layer2_perceptrons)

	#print(hidden_layer2_perceptrons)
	# activation inputs for hidden layer 1
	activation_inputs_2 = []
	for indx, hidden_layer2_perceptron in enumerate(hidden_layer2_perceptrons):
		activation_input = np.dot(activation_outputs_1,hidden_layer2_perceptron)
		activation_inputs_2.append(activation_input[0])

	activation_inputs_2 = np.array(activation_inputs_2)

	# Outputs for each perceptron in hidden layer 1
	# after sigmoid has been applied
	activation_outputs_2 = []
	for activation_input in activation_inputs_2:
		activation_output = sigmoid(activation_input)
		activation_output_temp = []
		activation_output_temp.append(activation_output)
		activation_outputs_2.append(activation_output_temp)

	activation_outputs_2 = np.array(activation_outputs_2)
	activation_outputs_2 = np.transpose(activation_outputs_2)

	######################################################################################################################
	# Ouput Layer

	# Get random weights from hidden layer to output perceptron
	random_weights = np.random.rand(5,1)
	output_layer_perceptrons = random_weights[:]

	output_layer_perceptrons = np.array(output_layer_perceptrons)

	activation_input_out = np.dot(activation_outputs_2, output_layer_perceptrons)
	activation_output_out = sigmoid(activation_input_out)

	######################################################################################################################
	cost = label[current_training_example][0] - activation_output_out

	######################################################################################################################

	print("Activation output: " + str(activation_output_out))
	print("Label: " + str(labels[index][0]))

	######################################################################################################################
	# Loss function - binary cross entropy loss function

	# If prediction is near correct for label then
	# the loss will be low
	# Otherwise if the prediction is far from correct 
	# based on the label, then loss will be high
	# Cost goes to infinity if label = 0 and prediction approaches 1
	# Cost goes to infinity if label = 1 and prediction approaches 0
	if labels[index][0] == 1:
		loss = -(math.log(activation_output_out))
	elif labels[index][0] == 0:
		loss = -(math.log(1-activation_output_out))

	print("Loss: " + str(loss))

	# this means predicted 1 and label is 1 meaning true positive
	if activation_output_out > 0.5 and labels[index][0] == 1:
		true_positives += 1
	# predicted 1 and label is 0 meaning false positive
	elif activation_output_out > 0.5 and labels[index][0] == 0:
		false_positives += 1
	elif activation_output_out < 0.5 and labels[index][0] == 1:
	# predicted 0 and label is 1 meaning false negative
		false_negatives += 1
	# predicted 0 and label is 0 meaning true negative
	elif activation_output_out < 0.5 and labels[index][0] == 0:
		true_negatives += 1

	accuracy = (true_positives+true_negatives)/(true_positives+true_negatives
		+false_positives+false_negatives)
	print("Accuracy: " + str(accuracy))
#print(labels)

'''
######################################################################################################################
'''
print(hidden_layer1_perceptrons[1])
print(training_examples_scaled[1])
print(hidden_layer1_perceptrons[1]*training_examples_scaled[1])
'''
#print(training_examples)


# model 1
'''
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

'''





