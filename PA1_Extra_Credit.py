# Tobby Lie
# CSCI-5931 PA1_Grad
# August 31, 2019
# Last modified: 9/9/19 @ 11:11AM

from sklearn import preprocessing
from random import random
from random import seed
import math
import numpy as np
import csv
import time

# to print entire list with out dots
import sys
np.set_printoptions(threshold=sys.maxsize)
######################################################################################################################
def listOfLists(lst):
	return [[el] for el in lst]

def sigmoid(x):
	if x < 0:
		return 1 - 1/(1 + math.exp(x))
	else:
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
labels_scaled = preprocessing.scale(labels)
######################################################################################################################
accuracies = []
precisions = []
recalls = []
f1_scores = []
######################################################################################################################
# ANN with back propogation
learning_rate = 0.5
true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

compare_accuracy = []
comapre_loss = []

# input layer 8 x 1
input_layer = np.random.rand(8, 1)
# weights from input layer to hidden layer
# intiialize as random
# 8x3
weights_1 = np.random.rand(len(input_layer), 3)
# weights form hidden layer to output layer
# initialize as random
# 3x1
weights_2 = np.random.rand(3, 1)
# output layer 1x1
output_layer = np.random.rand(1, 1)

# can comment this out if you want to use labels form dataset with 1's and 0's
# this is to test convergence with continuous values
labels = np.random.rand(7200, 1)

for index, label in enumerate(labels_scaled):
	# need to re-initialize input_layer so it maintains shape
	input_layer = np.random.rand(8, 1)
	# set input_layer equal to training_example set
	for indx, training_example in enumerate(training_examples_scaled[index]):
		input_layer[indx, 0] = training_example
	#print(input_layer.shape)
	# weighted sum result in hidden layer
	# 3x1
	# transpose input_layer in order to allow it to be matrix multiplied
	input_layer = input_layer.transpose()
	weighted_sum1s = np.matmul(input_layer, weights_1)
	weighted_sum1s = weighted_sum1s.transpose()
	# activation function in hidden layer (output_hl)
	# 3x1
	activation_1 = np.random.rand(3, 1)
	#print(weights_1.shape)
	#print(input_layer.shape)
	for indx, weighted_sum in enumerate(weighted_sum1s):
		activation_1[indx] = sigmoid(weighted_sum[0])
	# weighted sum result in output layer
	# 1x1
	# transpose activation_1 to matmul with weights_2
	activation_1 = activation_1.transpose()
	weighted_sums2 = np.matmul(activation_1, weights_2)
	# activation function in output layer (output_o)
	# 1x1
	activation_2 = np.random.rand(1, 1)
	activation_2[0] = sigmoid(weighted_sums2[0])

	# calculate error 
	# error = (1/2)*(target - output)^2
	error_out = (1/2)*(label-activation_2[0])
	error_out = error_out**2

	# calculate delta for weights connected to output layer
	# delta_output = (output - target)*(output)*(1 - output)
	# also change in error with respect to weight_n
	# = delta_output * activation_2_(connecting to weight n)

	delta_output = (activation_2[0] - label) * (activation_2[0]) * (1 - activation_2[0])
	# update weights connecting hidden layer to output layer
	for indx, weight_2 in enumerate(weights_2):
		weight_2 = weight_2 - (learning_rate)*(delta_output)*(activation_2[0])


	# for each perceptron in hidden layer, get weighted sum of delta_output
	# and all weights connecting to that perceptron
	# then multiply that weighted sum to activation in that hidden layer perceptron
	# multiplied by (1 - that activation) multiplied by the input connected
	# to that perceptron by the weight to be updated
	delta_hidden = np.random.rand(3, 1)
	for x in range(0, 3):
		delta_hidden[x, 0] = activation_2[0] * weights_2[x, 0]
	# update each weight connecting the input layer to the hidden layer
	# loop through each input node's 3 weights
	for x in range(0, 8):
		for y in range(0, 3):
			weights_1[x, y] = weights_1[x, y] - ((learning_rate)*(delta_hidden[y, 0]
			*(activation_1[0, y])*(1-activation_1[0, y])*(input_layer[0, x])))

	#print("-------------------------------------")
	# this means predicted 1 and label is 1 meaning true positive
	'''if activation_2[0] > 0.5 and label == 1:
		print("here")
		true_positives += 1
	# predicted 1 and label is 0 meaning false positive
	elif activation_2[0] > 0.5 and label == 0:
		false_positives += 1
	elif activation_2[0] < 0.5 and label == 1:
	# predicted 0 and label is 1 meaning false negative
		false_negatives += 1
	# predicted 0 and label is 0 meaning true negative
	elif activation_2[0] < 0.5 and label == 0:
		true_negatives += 1'''

	# can comment this out and uncomment block above to test for discrete
	# values of 1's and 0's
	# this block below is for continuous values to test for convergence
	if activation_2[0] > 0.5 and label > 0.5:
		true_positives += 1
	# predicted 1 and label is 0 meaning false positive
	elif activation_2[0] > 0.5 and label < 0.5:
		false_positives += 1
	elif activation_2[0] < 0.5 and label > 0.5:
	# predicted 0 and label is 1 meaning false negative
		false_negatives += 1
	# predicted 0 and label is 0 meaning true negative
	elif activation_2[0] < 0.5 and label < 0.5:
		true_negatives += 1

	accuracy = (true_positives+true_negatives)/(true_positives+true_negatives
		+false_positives+false_negatives+.000000001)

	# Loss function - binary cross entropy loss function

	# If prediction is near correct for label then
	# the loss will be low
	# Otherwise if the prediction is far from correct 
	# based on the label, then loss will be high
	# Cost goes to infinity if label = 0 and prediction approaches 1
	# Cost goes to infinity if label = 1 and prediction approaches 0
	if label == 1:
		loss = -(math.log(activation_2[0]))
	elif label == 0:
		loss = -(math.log(1-activation_2[0]))

	# print metrics
	print("-------------------------------------")
	print("Accuracy: " + str(accuracy))
	print("Loss: " + str(error_out))
	print(label)
	print("-------------------------------------")

	if index == 0 or index == len(labels)-1:
		compare_accuracy.append(accuracy)
		comapre_loss.append(error_out)

# print only first and last metrics to compare
print("Here are the first and last metrics 0 - first, 1 - last:")
for indx, acc in enumerate(compare_accuracy):
	print(str(indx))
	print("-------------------------------------")
	print("Accuracy: " + str(acc))
	print("Loss: " + str(comapre_loss[indx]))
	print("-------------------------------------")