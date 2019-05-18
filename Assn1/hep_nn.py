#DAT112 Assignment 1
#Wennie Wu
#02MAY2019

#This script retrieves the HEPMASS dataset and aims to develop a neural network that is able to categorize the data based on the label number
import pandas as pd 
import keras.models as km 
import keras.layers as kl
import keras.utils as ku
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#import dataset 
print("Reading HEP file.")
url = "https://support.scinet.utoronto.ca/~ejspence/HEPMASS_small.csv"
HEPData  = pd.read_csv(url)

#Exclude mass column data, won't be using it and #label columns, which is our target column
inputData = HEPData.drop(columns = ['mass','# label'])

#Extract the # label column, this is the target column (ouput data)
#outputData is the data points we want the model to determine given a set of input values
outputData = HEPData[['# label']]

#split data into training and testing (80/20)
#train_test_split function will split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(inputData, outputData, test_size = 0.2)

#convert output data into "one hot encoding" format, we have 2 categories 0 or 1
y_train = ku.to_categorical(y_train,2)
y_test = ku.to_categorical(y_test,2)

#Build the neural network model to predict the label of the HEP events
print("Building network.")

# numnodes is the number of nodes in the hidden layer
# 2 possible output values, either 0 or 1
# input size indicates the number of features, default can be set to 27
# d_rate is the dropout rate to account for over fitting of the training data, we will drop 40% of the neurons at random
def get_model(numnodes, d_rate = 0.3, input_size = 27, output_size = 2):
	#initialize the model.
	model = km.Sequential()

	#add a hidden layer (fully connected) with the specified number of nodes
	model.add(kl.Dense(numnodes, name = 'hidden', input_dim = input_size, activation = 'tanh'))

	#add a dropout step to drop 40% of the nodes to sample from 
	model.add(kl.Dropout(d_rate, name = 'dropout'))

	#add batch normalization of the data
	model.add(kl.BatchNormalization(name = 'batch_normalization'))
	
	#add an output layer
	model.add(kl.Dense(output_size, activation = 'softmax', name = 'output'))

	return model


#Design the model with the indicated number of neurons/nodes in the hidden layer
model = get_model(30)

#Begin training the network with the training data
print("Training network.")

#Compile the neural network
#optimization algorithm used will be stochastic gradient descent 
# lost/cost function to use is categorical cross entropy as our data is in categorical format
model.compile (optimizer = 'sgd', metrics = ['accuracy'], loss  = "categorical_crossentropy")

# Train the network on the training data
fit = model.fit(X_train,y_train, epochs = 50, batch_size = 25, verbose = 1)

# evaluates the network on the test data, print out the result
testScore = model.evaluate(X_test, y_test)
print("The test score is ", testScore)

#Summarize the history for loss as a function of Epoch
plt.plot(fit.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

