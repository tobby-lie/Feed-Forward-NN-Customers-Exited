Tobby Lie
CSCI-5931
9/11/19
PA1
######################################################
CUSTOMERS EXITED
######################################################
This program is a program that utilizes various feed-
forward neural networks in order to predict which 
customers from a European Bank will exit or stay 
(1 or 0).
######################################################
What's included in the .zip:
1. dataset.csv - preprocessed data file that will be 
used to train and test the data based on features and
labels.
2. judge-pred.csv - predictions gathered from trained
model output in .csv format with columns 'CustomerID
and Exited'
3. judge.csv -  data used to predict with trained model
with highest accuracy.
4. model_4.h5 - model 4 had the highest accuracy. .h5
file was used in order save and load the model.
5. Report.pdf - pdf of report specifying structure of
models utilized and their metrics.
6. test-code.py - .py file used to predict using saved
model.
7. Training-code.py - .py file used to train and test
various models.
######################################################
How to TRAIN:
In terminal, once in directory with .zip contents
run with python3 training-code.py.
Metrics at each epoch will be displayed for training
data and at the end test metrics will be displayed.
######################################################
How to TEST:
In terminal, once in directory with .zip contents
run with python3 test-code.py.
judge-pred.csv will populate with the predictions
for each customerID.
######################################################
