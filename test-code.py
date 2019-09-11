# Tobby Lie
# CSCI-5931 PA1
# August 31, 2019
# Last modified: 9/11/19 @ 5:26PM

import csv
import numpy as np
from sklearn import preprocessing
from keras.models import model_from_json   

def load_model():
	'''loads saved model for predicting'''
	global model

	json_file = open('model.json', 'r')
	model_json = json_file.read()
	model = model_from_json(model_json)
	model.load_weights("model_4.h5")

# opens judge.csv and extracts features into lists which then get added to a list
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
	# get all features and append to features list to create a vector
	for row in readCSV:
		# need to save cust_ID for later use when predicting
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
		# add features vector to test_model_examples
		# this is so it is a lists of lists to represent many examples
		#  with its own set of features
		test_model_examples.append(features[:])
		# clear features list each time to be used again
		features.clear()

# load model
load_model()

# turn list into np.array and feature scale with preprocessing
test_model_examples = np.array(test_model_examples)
test_model_examples = preprocessing.scale(test_model_examples)
# create list of predictions using .predict with keras
predictions = model.predict(test_model_examples)
# based on the probabilities of the predictions make decisions on whether
# output is 0 or 1
for index, prediction in enumerate(predictions):
	if prediction[0] < 0.5:
		prediction[0] = 0
	elif prediction[0] > 0.5:
		prediction[0] = 1
# create list of lists to hold each prediction pair (customer ID, prediction)
prediction_pairs = [[] for x in range(len(predictions))]
# append customer ID and prediction for each prediction_pairs entry
for index, prediction in enumerate(predictions):
	prediction_pairs[index].append(cust_ID[index])
	prediction_pairs[index].append(prediction[0])
# make prediction_pairs an np.array in order to be written out to file
prediction_pairs = np.array(prediction_pairs)
# save results to file
np.savetxt("judge-pred.csv", prediction_pairs, delimiter=",", fmt=("%i, %i"))