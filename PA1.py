# Tobby Lie
# CSCI-5931 PA1
# August 31, 2019

from keras.models import Sequential
from keras.layers import Dense, Activation
import keras_metrics
import numpy as np
import csv

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

	features = []
	training_examples = []
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

		if label == 'Exited':
			continue

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

'''
data = np.random.random((1000, 100))
print(len(training_examples[0]))
'''

'''
labels = np.random.random_sample((3,4))
print(labels)
print(labels[1:2,1:2])
'''

training_examples = np.array(training_examples)
labels = np.array(labels)

# model 1

model_1 = Sequential()
model_1.add(Dense(10, activation='relu', input_dim=8))
model_1.add(Dense(5, activation='relu'))
model_1.add(Dense(4, activation='relu'))
model_1.add(Dense(1, activation='sigmoid'))
model_1.compile(optimizer='rmsprop',
			loss='binary_crossentropy',
			metrics=['accuracy'])
model_1.fit(training_examples, labels, epochs=100, batch_size=30)

model_2 = Sequential()
model_2.add(Dense(11, activation='relu', input_dim=8))
model_2.add(Dense(9, activation='relu'))
model_2.add(Dense(5, activation='relu'))
model_2.add(Dense(3, activation='sigmoid'))
model_2.add(Dense(1, activation='sigmoid'))
model_2.compile(optimizer='rmsprop',
			loss='binary_crossentropy',
			metrics=['accuracy'])
model_2.fit(training_examples, labels, epochs=100000, batch_size=30)

model.summary()