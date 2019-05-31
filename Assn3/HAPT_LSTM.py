#DAT112 Assignment 3
#Wennie Wu
#30MAY2019

#This script retrieves the HAPT dataset and aims to create a Long short-term memory recurrent neural network architecture to 
# classify time series data

import pandas as pd 
import numpy as np
import keras.models as km 
import keras.layers as kl
import keras.utils as ku
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#import dataset 
print("Reading HAPT file.")
#url = "https://support.scinet.utoronto.ca/~ejspence/HAPT.data.csv"
file = "./subhapt2.csv"
HAPTData  = pd.read_csv(file)

#Input data is the time series data
chunk_size = 50 
time_series_50 = []
output_50 = []
num_features = HAPTData.shape[1] - 1 #to account for ID column

#Separate the data into 50 categorical chunks (same category time sequence)
i = 0
while i <= (len(HAPTData) - chunk_size):
	#Get the action category of the first value of the chunk
	id_val = HAPTData.loc[i,'ID']
	temp_50_chunk = HAPTData.loc[i:i+chunk_size -1]
#Check if all values in the chunk are the same as the first value in the chunk 
	if(temp_50_chunk['ID'].all() == id_val.all()):
		#The last value in this chunk has the same action category as the first value in the chunk 
		#Add to time series dataframe 
		inputdf_to_list = temp_50_chunk.drop(columns = 'ID')
		inputdf_to_list = inputdf_to_list.to_numpy()
		time_series_50.append(inputdf_to_list)
		outputdf_to_list = temp_50_chunk['ID'].to_numpy()
		output_50.append(outputdf_to_list)
		i += 2
	else:
                #This 50 chunk is not all the same, determine where the new category starts, set index to that value
		temp = HAPTData.loc[i:i+chunk_size - 1]
		num_rows_to_drop = temp.loc[temp['ID'] == id_val].shape[0]
		i += num_rows_to_drop


#split data into training and testing (70/30) turn randomizer off, we don't want the data to be randomized
X_train_list, X_test_list, y_train_list, y_test_list = train_test_split(time_series_50, output_50, test_size = 0.3, random_state = 0)

print("Reshaping the data.")

#reshape the data 
#unravel the list of time series so that we can reshape it to be 3D as the LSTM model expects a 3D input
#unravel the list of values first, then reshape it
X_train_unravel = np.concatenate(X_train_list).ravel().ravel()
#convert from 1D to 3D for keras LSTM 
X_train = np.reshape(X_train_unravel,(int(len(X_train_unravel)/num_features),1,num_features))

#Do the same for the test data set
X_test_unravel = np.concatenate(X_test_list).ravel().ravel()
X_test = np.reshape(X_test_unravel,(int(len(X_test_unravel)/num_features),1,num_features))


#format the output values
y_train_unravel = np.concatenate(y_train_list).ravel()
#convert output data into "one hot encoding" format, we have 12 categories
y_train = ku.to_categorical(y_train_unravel)

y_test_unravel = np.concatenate(y_test_list).ravel()
y_test = ku.to_categorical(y_test_unravel)

#Build the model
print("Building the LSTM network.")
model = km.Sequential()
model.add(kl.LSTM(200, input_shape = (X_train.shape[1], num_features)))
#add a dropout step to drop 50% of the nodes to sample from
model.add(kl.Dropout(0.4))
#add a fully connected layer
model.add(kl.Dense(100, activation = 'relu')) 
model.add(kl.Dense(13, activation = 'softmax'))

#Compile the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print("Training the network.")
#fit the data to the model! 
fit = model.fit(X_train, y_train, epochs = 50, batch_size = 10, verbose = 2)

# evaluates the network on the test data, print out the result
testScore = model.evaluate(X_test, y_test)
print("The test score is ", testScore)

# # Save the model so that we can use it as a starting point.
# model.save('data/' + modelfile)
#Summarize the history for loss as a function of Epoch
plt.plot(fit.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
