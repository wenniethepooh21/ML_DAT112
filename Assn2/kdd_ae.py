#DAT112: Assignment 2
#Wennie Wu
#MAY162019

#This script creates an autoencoder that is able to detect malicious logins 

#import necessary libraries to build autoencoder and display summary of its output
import pandas as pd 
import numpy as np
import keras.models as km 
import keras.layers as kl
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix

#import dataset from URL
print("Reading KDD data file.")
url = "https://support.scinet.utoronto.ca/~ejspence/Ass2.kddcup.csv"
KDD = pd.read_csv(url)

#Divide data into training, validation and testing datasets (60%,40%,40%)

Train, Val, Test = np.split(KDD.sample(frac = 1), [int(0.6*len(KDD)), int(0.8*len(KDD))])

#remove the malicious inputs in the training data set
Train = Train[Train['label'] == 0]

#drop the label column from the training dataset
Train = Train.drop(columns = ['label'])

#extract the label column / correct output from validation and test datasets
Val_label = Val['label']
Test_label = Test['label']

#drop the label column from validation and test datasets
Val = Val.drop(columns = ['label'])
Test = Test.drop(columns = ['label'])

#Transforms the features in a range from 0 -1 (scales the data)
Train = minmax_scale(Train)
Val = minmax_scale(Val)
Test = minmax_scale(Test)

#Build an autoencoder using Keras
print("Building network.")

# #Find the number of columns, represents features 
input_dim = Train.shape[1]

#Build feedforward autoencoder
ae = km.Sequential()
#Encoder layers
ae.add(kl.Dense(25, activation = 'elu', input_shape = (input_dim,)))
ae.add(kl.Dense(16, activation = 'elu'))
ae.add(kl.Dense(4, activation = 'linear', name = 'bottleneck')) #This layer contains the compressed representation of the input data ('bottleneck layer')

#Deocder layers
ae.add(kl.Dense(16, activation = 'elu'))
ae.add(kl.Dense(25, activation = 'elu'))
ae.add(kl.Dense(34, activation = 'sigmoid'))

# output dimension of compressed layer from encoder
encoding_output_dim = 4

#specify the size of the input data to the decoder or the size of the output from the encoder
decoder_input = km.Input(shape = (encoding_output_dim,))

#first layer of the decoder, pass in the size of the output of the encoder to the decoder
decoder = ae.layers[-3](decoder_input) 

#second layer of the decoder, pass in the output from the first layer of the decoder
decoder = ae.layers[-2](decoder)

#third and last layer of the decoder, pass in the output from the second layer of the decoder
decoder = ae.layers[-1](decoder)


# build the encoder model with the autoencoder as the input and the output is the compressed data
encoder = km.Model(ae.input, ae.get_layer('bottleneck').output) # encoder output

#Build the decoder model given the input as the encoder output, and the output of the decoder
decoder = km.Model(inputs = decoder_input, outputs = decoder) #decoder output


#Compile the model with mean squared error as loss function, stochastic gradient descent as optimizer
ae.compile(loss="mean_squared_error", optimizer = 'sgd', metrics = ['accuracy'])

#Train the model on training data
print("Training network.")
ae.fit(Train,Train, batch_size = 25, epochs = 100, verbose = 0)

# Feed in validation data into the model to calculate the threshold
# calculate the mean square error for each data point in Validation data that's passed through the network
val_ae = ae.predict(Val)
#calculate mean squared error of the data points from validation dataset
val_mse = np.mean(np.power(val_ae - Val,2),1)

#calculates the mean and standard deviation of the normal data points, and the malicious data points
val_norm_mean = np.mean(val_mse[Val_label == 0])
val_norm_std = np.std(val_mse[Val_label==0])

#malicious validation data points 
val_mal_mean = np.mean(val_mse[Val_label == 1])
val_mal_std = np.std(val_mse[Val_label==1])

#print output
print("")
print("Summary of validation data results:")
print("binary_labels mean std")
print("0.0", val_norm_mean, val_norm_std)
print("1.0", val_mal_mean, val_mal_std)

#if the reconstruction error is > 0.05, login attempt is considered an anomaly 
print("")
print("Setting threshold to 0.05")
print("")

# Feed testing data through network to see how it performs
test_ae = ae.predict(Test)
mse = np.mean(np.power(test_ae - Test,2),1)

predict_malicious = np.zeros(len(Test_label))
predict_malicious[np.nonzero(mse > 0.05)] = 1

#print output of how well network was at classifying the given dataset with a confusion matrix
print("Test data confusion matrix: ")
conf_matrix = confusion_matrix(np.array(Test_label), predict_malicious)
print(conf_matrix)


















