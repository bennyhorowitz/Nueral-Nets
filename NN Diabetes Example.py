import os
#os.getcwd()
#os.chdir(r"C:\Users\horowitz-b\Documents\Spring 2021\BUAD 689 Deploying Models & Building Optimization models\Week 1 - Deploying Models in Python Code (Johnston)\Week 1 Assignment")

import pandas as pd
import numpy as np
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import csv

# load the dataset
dataset = loadtxt(r"C:\Users\horowitz-b\Documents\Spring 2021\BUAD 689 Deploying Models and Building Optimization models\Week 1 - Deploying Models in Python Code (Johnston)\Week 1 Assignment\diabetes.csv", delimiter=',', skiprows=1)
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

#Save the model
model.save('NNmodel.h5')

################################################################################

#Load the model
from keras.models import load_model
model = load_model('NNmodel.h5')

# load the new data that needs predictions
dataset2 = loadtxt('diabetes_new_data.csv', delimiter=',', skiprows=1)
# split into input (X) and output (y) variables

Xs = dataset2[:,0:8]

# make class predictions with the model
predictions2 = model.predict_classes(Xs)

#Load the predicted values into a lis
pred = []
for i in range(len(dataset2)):
  pred.append(predictions2)

#Load the list of predictions to a csv file
with open('predictionData.csv', 'w', newline='',) as f:
    writer = csv.writer(f)
    for val in predictions2:
        writer.writerow(val)
        
#Create pandas dataframes out of new data csv and preditions csv
a = pd.read_csv("predictionData.csv",header=None)
a=pd.DataFrame(a.values, columns = ['Prediction'])
b = pd.read_csv("diabetes_new_data.csv")

#Add the columns from the predictio
result = pd.concat([b, a], axis=1)

result.to_csv(r'diabetes-predictions.csv', index = False)    


