#!/usr/bin/env python3
import numpy as np
from dense_layer import dense_layer


class FFNN(): # neural network class
    def __init__(self, learning_rate = 0.1, batch_size = 0.1, targetAccuracy = 0.9):
        self.learning_rate = learning_rate
        self.batch_size = batch_size # share of the dataset used to update params
        self.layers = []
        self.targetAccuracy = targetAccuracy # when this accuracy (plus sufficient convergence) is achieved, training is halted\\

    def addLayer(self, layer):
        self.layers.append(layer)

    def __construct(self, d, k = 1): # private constructor method to create layers and initialize params
        # d is number of features
        # k is number of classes in classification

        self.layers.append(DenseLayer(k, 'output')) # add output layer
        self.layers = np.array(self.layers) # convert layer list to Numpy array

        #initialize first layer
        self.layers[0].construct(np.random.rand(self.layers[0].size, d) - 0.5,
                                 np.random.rand(self.layers[0].size)-0.5)

        for h in range(1, len(self.layers)): # create hidden layers
            self.layers[h].construct(
                np.random.rand(self.layers[h].size,
                               self.layers[h-1].size) - 0.5,
                np.random.rand(self.layers[h].size) - 0.5)

    def train(self, x, y):
        iters = 0
        eta = self.learning_rate
        self.__construct(x.shape[1], len(np.unique(y))) # call internal constructor method

        data = np.concatenate((x, y.reshape(y.size, 1)), 1) # merge x and y data such that the data/labels correspond
        batchSize = int(self.batch_size * x.shape[0])

        while(not self.__earlyStopping(data,iters)):
            # randomly select batch to train on
            index = np.random.randint(x.shape[0], size = batchSize) # create random index set to sample
            batch = data[index,:] # create random sample from the data

            for sample in range(batchSize): # iterate through all samples in the batch
                point = batch[sample, :-1]
                label = batch[sample, -1]
                self.__forwardPass(point, label) # call internal prediction method

                # compute derivative with respect to nodes in last layer
                self.layers[-1].derivatives = (
                    self.layers[-1].values - self.__oneHot(label))/batchSize
                self.layers[-1].adjustParams(self.layers[-2].values, eta)

                for i in range(len(self.layers)-2, -1, -1):
                    self.layers[i].differentiateDense(
                        self.layers[i+1].w,
                        self.layers[i+1].derivatives.reshape(self.layers[i+1].size, 1))
                    if i==0: self.layers[0].adjustParams(point, eta)
                    else: self.layers[i].adjustParams(self.layers[i-1].values, eta)

            iters += 1
            print('iterations: ', iters)

    def __oneHot(self, x):
        vect = np.zeros(self.layers[-1].size)
        vect[int(x)] = 1
        return vect

    def __forwardPass(self, x, y): # accepts x vector and y label (forward pass for one label)
        self.layers[0].compute(x)
        for i in range(1, self.layers.size): # compute layer values as matmuls
            self.layers[i].compute(self.layers[i-1].values)
        return np.argmax(self.layers[-1].values) # predict the maximum value in the output layer

    def __softmax(self, x):
        maxi = np.max(x) # normalize the values to prevent overflows
        sumAll = sum(np.exp(x-maxi))
        return np.exp(x-maxi)/sumAll
def predict(self, x, y): # public prediction method
        preds = np.empty(x.shape[0],int)
        for i in range(x.shape[0]):
            preds[i] = self.__forwardPass(x[i], y[i])
        acc = self.__accuracy(preds, y)
        return preds, acc

    def __earlyStopping(self, data, iteration): # decides when to stop the training\n
        global oldAccuracy
        if (iteration % 10 != 0): return False
        if iteration == 0: # if it's the first iterati skip everything here right away
            oldAccuracy = 0.3 # set accuracy to 30%
            return False
        else: # check only every 10'th iteration
            index = np.random.randint(data.shape[0], size = int(data.shape[0]*0.2)) # create random index set to sample
            sample = data[index,:] # create random sample from the data
            predictions = self.predict(sample[:,:-1], sample[:,-1])[0] # predict on the sample
            accuracy = self.__accuracy(predictions, sample[:, -1])
            print('current accuracy: ', accuracy)
            # if accuracy barely changes and achieved target accuracy, stop training
            if ((abs(accuracy - oldAccuracy) < 0.01) &
                (accuracy > self.targetAccuracy)):
                return True
            oldAccuracy = accuracy
            return False

    def __accuracy(self, preds, y):
        if (y.size == 0):
            raise Exception('empty sample')
        errors = 0
        for i in range(y.size):
            if (int(preds[i]) != int(y[i])):
                errors += 1
        return 1 - errors/y.size


# class FFNN(): # neural network class

#     def __init__(self, learning_rate=0.1, batch_size=0.1, targetAccuracy=0.9):
#         self.learning_rate = learning_rate
#         self.batch_size = batch_size  # share of the dataset used to update params
#         self.layers = [Dense_layer(None, "input")]  # initialize an input layer
#         self.targetAccuracy = targetAccuracy  # when this accuracy (plus sufficient convergence) is achieved, training is halted\

#     def addLayer(self, layer):
#         self.layers.append(layer)

#     def __construct(self, d, k = 1): # private constructor method to create layers and initialize params
#         # d is number of features
#         # k is number of classes in classification

#         self.layers.append(Dense_layer(k, "output")) # add output layer
#         self.layers = np.array(self.layers) # convert layer list to Numpy array

#         #initialize input layer
#         self.layers[0].size = d
#         self.layers[0].construct(None, np.random.rand(d) - 0.5) # initialize to random b vector

#         for h in range(1, len(self.layers)): # create hidden layers
#             self.layers[h].construct(np.random.rand(self.layers[h].size, self.layers[h-1].size) - 0.5, np.random.rand(self.layers[h].size) - 0.5)

#     def train(self, x, y):
#         iters = 0
#         eta = self.learning_rate
#         self.__construct(x.shape[1], len(np.unique(y))) # call internal constructor method

#         data = np.concatenate((x, y), 1) # merge x and y data such that the data/labels correspond
#         batchSize = int(self.batch_size * x.shape[0])

#         while(not self.__earlyStopping(x,y,iters)):
#             # randomly select batch to train on
#             index = np.random.randint(x.shape[0], size = batchSize) # create random index set to sample
#             batch = data[index,:] # create random sample from the data

#             for sample in range(batchSize): # iterate through all samples in the batch
#                 point = batch[sample, :-1]
#                 label = batch[sample, -1]
#                 self.__forwardPass(point, label) # call internal prediction method

#                 # compute derivative with respect to nodes in last layer
#                 self.layers[-1].derivatives = (self.layers[-1].values - self.__oneHot(label))/batchSize
#                 self.layers[-1].adjustParams(self.layers[-2].values, eta)

#                 for i in range(len(self.layers)-2, 0, -1):
#                     self.layers[i].differentiateDense(self.layers[i+1].w, self.layers[i+1].derivatives.reshape(self.layers[i+1].size, 1))
#                     self.layers[i].adjustParams(self.layers[i-1].values, eta)

#             iters += 1
#             print("iterations: ", iters)

#     def __oneHot(self, x):
#         vect = np.zeros(self.layers[-1].size)
#         vect[int(x)] = 1
#         return vect

#     def __forwardPass(self, x, y): # accepts x vector and y label (forward pass for one label)
#         self.layers[0].values = x
#         for i in range(1, self.layers.size): # compute layer values as matmuls
#             self.layers[i].compute(self.layers[i-1].values)
#         return np.argmax(self.layers[-1].values) # predict the maximum value in the output layer

#     def __softmax(self, x):
#         maxi = np.max(x) # normalize the values to prevent overflows
#         sumAll = sum(np.exp(x-maxi))
#         return np.exp(x-maxi)/sumAll

#     def predict(self, x, y): # public prediction method
#         preds = np.empty(x.shape[0],int)
#         for i in range(x.shape[0]):
#             preds[i] = self.__forwardPass(x[i], y[i])
#         acc = self.__accuracy(preds, y)
#         return preds, acc

#     def __earlyStopping(self, x, y, iteration): # decides when to stop the training
#         global oldAccuracy
#         if (iteration % 10 != 0): return False
#         if iteration == 0: # if it's the first iteration, skip everything here right away
#             oldAccuracy = 0.3 # set accuracy to 30%
#             return False
#         else: # check only every 10'th iteration
#             data = np.concatenate((x, y), 1) # merge x and y so that they correspond (to sample from it)
#             index = np.random.randint(x.shape[0], size = int(x.shape[0]*0.2)) # create random index set to sample
#             sample = data[index,:] # create random sample from the data
#             predictions = self.predict(sample[:,:-1], sample[:,-1])[0] # predict on the sample
#             accuracy = self.__accuracy(predictions, sample[:, -1])
#             print("current accuracy: ", accuracy)

#             # if accuracy barely changes and achieved target accuracy, stop training
#             if ((abs(accuracy - oldAccuracy) < 0.01) & (accuracy > self.targetAccuracy)): return True
#             oldAccuracy = accuracy
#             return False

#     def __accuracy(self, preds, y):
#         if (y.size == 0): raise Exception("empty sample")
#         errors = 0
#         for i in range (y.size):
#             if (preds[i] != int(y[i])): errors += 1
#         return 1 - errors/y.size


