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
url = "https://support.scinet.utoronto.ca/~ejspence/HAPT.data.csv"
HAPTData  = pd.read_csv(url)

#Input data is the time series data
chunk_size = 50 
time_series_50 = []
output_50 = []
num_features = HAPTData.shape[1] - 1 #to account for ID column

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
		temp = HAPTData.loc[i:i+chunk_size - 1]

		num_rows_to_drop = temp.loc[temp['ID'] == id_val].shape[0]
		i += num_rows_to_drop




# #Iterate through the HAPTData set to separate into chunks of 50, to overlap time series, shift down 2
# for i in range(0,len(HAPTData)-chunk_size,2):
# 	#Get the action category of the first value of the chunk
# 	id_val = HAPTData.loc[i,'ID']

# #Check if the 50th value in the chunk is the same as the first value of the same chunk
# 	if(HAPTData.loc[i+chunk_size -1,'ID'] == id_val):
# 		#The last value in this chunk has the same action category as the first value in the chunk 
# 		#Add to time series dataframe 
# 		inputdf_to_list = HAPTData.loc[i:i+chunk_size-1].drop(columns = 'ID')
# 		inputdf_to_list = inputdf_to_list.to_numpy()
# 		time_series_50.append(inputdf_to_list)

# 		outputdf_to_list = HAPTData.loc[i:i+chunk_size - 1]['ID'].to_numpy()
# 		output_50.append(outputdf_to_list)
# 	else:
# 		#The last value in this chunk does not have the same action category as the first value
# 		#drop the rows that don't have the same action category, it will be in the next chunk 
# 		temp = HAPTData.loc[i:i+chunk_size - 1]
# 		# num_rows_to_drop = temp.loc[temp['ID'] != id_val].shape[0]
# 		# inputdf_to_list = HAPTData.loc[i:i+chunk_size- 1 -num_rows_to_drop].drop(columns = 'ID')
# 		# inputdf_to_list = inputdf_to_list.to_numpy()
# 		# time_series_50.append(inputdf_to_list)

# 		# outputdf_to_list = HAPTData.loc[i:i+chunk_size- 1 -num_rows_to_drop]['ID'].to_numpy()
# 		# output_50.append(outputdf_to_list)

# 		# i += len(inputdf_to_list)


# 		num_rows_to_drop = temp.loc[temp['ID'] == id_val].shape[0]
# 		i += num_rows_to_drop -2


#split data into training and testing (70/30)
#train_test_split function will split the data into training and testing data
X_train_list, X_test_list, y_train_list, y_test_list = train_test_split(time_series_50, output_50, test_size = 0.3)

print("Reshaping the data.")

#reshape the data 
train_num_50_chunks = len(X_train_list)

#unravel the list of time series so that we can reshape it to be 3D as the LSTM model expects a 3D x input
X_train_unravel = np.concatenate(X_train_list).ravel()
X_train = np.reshape(X_train_unravel,(train_num_50_chunks,chunk_size,num_features))

test_num_50_chunks = len(X_test_list)
X_test_unravel = np.concatenate(X_test_list).ravel()
X_test = np.reshape(X_test_unravel,(test_num_50_chunks,chunk_size,num_features))

y_train_unravel = np.concatenate(y_train_list).ravel()
y_train = np.reshape(y_train_unravel, (train_num_50_chunks, chunk_size))

y_test_unravel = np.concatenate(y_test_list).ravel()
y_test = np.reshape(y_test_unravel, (test_num_50_chunks, chunk_size))


print("Building the LSTM network.")

model = km.Sequential()
model.add(kl.LSTM(100, input_shape = (chunk_size, num_features)))
model.add(kl.Dense(chunk_size, activation = 'softmax'))

print("Training the network.")
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#fit the data to the model 
fit = model.fit(X_train, y_train, epochs = 50, batch_size = 50, verbose = 2)

# evaluates the network on the test data, print out the result
testScore = model.evaluate(X_test, y_test)
print("The test score is ", testScore)

# # Save the model so that we can use it as a starting point.
# model.save('data/' + modelfile)
