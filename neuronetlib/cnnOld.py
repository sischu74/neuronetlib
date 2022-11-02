#!/usr/bin/env python3
import numpy as np
from dense_layer import Dense_layer
from conv_layer import Conv_layer
from pool_layer import Pool_layer

# class CNN(): # convolutional neural network class
#     def __init__(self, learning_rate = 0.1, batch_size = 0.1, targetAccuracy = 0.9):
#         self.layers = []
#         self.eta = learning_rate
#         self.batch_size = batch_size
#         self.targetAccuracy = targetAccuracy

#     def addLayer(self, layer):
#         self.layers.append(layer)

#     def __construct(self, x, k: int) -> None:
#         # x is a sample input (used for input size), k is number of output classes
#         self.layers.append(Dense_layer(k, "output")) # add output layer
#         self.layers = np.array(self.layers)
#         self.layers[-1].construct(None, np.random.rand(k) - 0.5)

#         # construct input layer
#         assert (type(self.layers[0]) != Dense_layer), "First layer cannot be dense."
#         self.layers[0].construct(x.shape)

#         for l in range(1, len(self.layers)):  # iterate hidden layers
#             if isinstance(self.layers[l], Dense_layer):
#                 self.layers[l].construct(np.random.rand(self.layers[l].size,
#                     self.layers[l-1].outputDim[0]*self.layers[l-1].outputDim[1])-0.5,
#                     np.random.rand(self.layers[l].size)-0.5)
#             else:
#                 self.layers[l].construct(self.layers[l-1].outputDim)

#     def train(self, x: np.ndarray, y: np.ndarray) -> None:
#         inputDim = x[0].shape
#         self.__construct(x[0], np.unique(y).size)
#         iters = 0

#         data = np.reshape(x, (x.shape[0], x[0].size))  # flatten array
#         data = np.concatenate((data, y.reshape((y.shape[0], 1))), 1)  # merge x and y data such that the data/labels correspond
#         batch_size = int(self.batch_size * x.shape[0])

#         while not self.__earlyStopping(x, y, iters):
#             # randomly select batch to train on
#             index = np.random.randint(x.shape[0], size=batch_size)  # create random index set to sample
#             batch = data[index, :]  # create random sample from the data

#             for sample in range(batch_size):  # iterate through all samples in the batch
#                 point = batch[sample, :-1]
#                 label = batch[sample, -1]
#                 self.__forwardPass(np.reshape(point, inputDim), label)  # call internal prediction method

#                 # compute derivative with respect to nodes in last layer
#                 self.layers[-1].derivatives = (self.layers[-1].values - self.__oneHot(label))/batch_size
#                 self.layers[-1].adjustParams(self.layers[-2].values, self.eta)

#                 for i in range(len(self.layers)-2, 0, -1):
#                     currLayer = self.layers[i]
#                     if isinstance(self.layers[i+1], Dense_layer):
#                         currLayer.differentiateDense(self.layers[i+1].w, self.layers[i+1].derivatives.reshape(self.layers[i+1].size, 1))
#                     elif isinstance(self.layers[i+1], Pool_layer):
#                         currLayer.differentiatePool(self.layers[i+1].pos)
#                     elif isinstance(self.layers[i+1], Conv_layer):
#                         currLayer.differentiateConv(self.layers[i+1].derivatives, self.layers[i+1].oldStencil)

#                     currLayer.adjustParams(self.layers[i-1].values, self.eta)

#             iters += 1
            # print(f"iterations: {iters}")

#     def __forwardPass(self, x, y):
#         self.layers[0].compute(x)
#         for i in range(1, self.layers):
#             self.layers[i].compute(self.layers[i-1].values)
#         return np.argmax(self.layers[-1].values) # predict the maximum value in the output layer

#     def predict(self, x, y):
#         newInput = np.empty((len(x), self.layers[-1].getSize()))
#         for i in range(len(x)): # iterate through samples
#             self.layers[0].compute(x[i])
#             for j in range(self.layers.size -1): # propagate the value through all layers
#                 self.layers[j+1].compute(self.layers[j].getValues())
#             newInput[i] = self.__vectorize(self.layers[-1].getValues()) # store value as input for ffnn

#         return self.ffnn.predict(newInput, y)

#     def __earlyStopping(self, x, y, iteration): # decides when to stop the training
#         if (iteration % 10 != 0): return False
#         if iteration == 0: # if it's the first iteration skip everything here right away
#             return False
#         else: # check only every 10'th iteration
#             data = np.concatenate((x, y), 1) # merge x and y so that they correspond (to sample from it)
#             index = np.random.randint(x.shape[0], size = int(x.shape[0]*0.2)) # create random index set to sample
#             sample = data[index,:] # create random sample from the data
#             predictions = self.predict(sample[:,:-1], sample[:,-1])[0] # predict on the sample
#             accuracy = self.__accuracy(predictions, sample[:, -1])
#             print("current accuracy: ", accuracy)

#             if accuracy > self.targetAccuracy:
#                 return True
#             return False








class CNN(): # convolutional neural network class
    def __init__(self, learning_rate = 0.1, batch_size = 0.1, targetAccuracy = 0.9):
        self.layers = []
        self.eta = learning_rate
        self.batch_size = batch_size
        self.targetAccuracy = targetAccuracy
        self.inputDim = None

    def addLayer(self, layer):
        self.layers.append(layer)

    def __construct(self, x, k):
        # x is a sample input (used for input size), k is number of output classes
        self.inputDim = list([1, x.shape[0], x.shape[1]])
        self.layers.append(Dense_layer(k, 'output')) # add output layer
        self.layers = np.array(self.layers)

        if(type(self.layers[0]) == Dense_layer):
            self.layers[0].construct(np.random.rand(self.layers[0].size, x.size)-0.5,
                                     np.random.rand(self.layers[0].size)-0.5)
        else: self.layers[0].construct((1, x.shape[0], x.shape[1]))

        first = True # indicates if current layer is the first dense hidden layer
        for l in range(1, len(self.layers)):
            if (type(self.layers[l]) == Dense_layer):
                if (first & (type(self.layers[l-1]) != Dense_layer)):
                    first = False
                    predDim = self.layers[l-1].outputDim # dimensions of last non-dense layer
                    self.layers[l].construct(np.random.rand(
                        self.layers[l].size, predDim[0]*predDim[1]*predDim[2]) - 0.5,
                                             np.random.rand(self.layers[l].size) - 0.5)
                else:
                    assert(type(self.layers[l-1]) == Dense_layer), 'Layer after dense layer must be dense too.'
                    self.layers[l].construct(np.random.rand(
                        self.layers[l].size, self.layers[l-1].size)-0.5,
                                             np.random.rand(self.layers[l].size)-0.5)
            else:
                self.layers[l].construct(self.layers[l-1].output_dim)

    def train(self, x, y):
        self.__construct(x[0], len(np.unique(y)))
        iters = 0

        data = np.reshape(x, (x.shape[0], x[0].size)) # flatten array
        data = np.concatenate((data, y.reshape(y.size,1)), 1) # merge x and y data such that the data/labels correspond
        batchSize = int(self.batch_size * x.shape[0])

        # for i in range(5):
        while not self.__earlyStopping(data, iters):
            # randomly select batch to train on
            index = np.random.randint(x.shape[0], size=batchSize) # create random index set to sample
            batch = data[index, :]  # create random sample from the data

            for sample in range(batchSize): # iterate through all samples in the batch
                point = batch[sample, :-1]
                label = batch[sample, -1]
                self.__forwardPass(np.reshape(point, self.inputDim), label) # call internal prediction method

                # compute derivative with respect to nodes in last layer
                self.layers[-1].derivatives = (
                    self.layers[-1].values - self.__oneHot(label))/batchSize
                self.layers[-1].adjustParams(self.layers[-2].values, self.eta)

                for i in range(len(self.layers)-2, -1, -1):
                    if isinstance(self.layers[i+1], Dense_layer):
                        self.layers[i].differentiateDense(
                            self.layers[i+1].w, self.layers[i+1].derivatives.reshape(
                                self.layers[i+1].size, 1))
                    elif isinstance(self.layers[i+1], Pool_layer):
                        self.layers[i].differentiatePool(
                            self.layers[i+1].fullDerMat, self.layers[i+1].pos)
                    elif isinstance(self.layers[i+1], Conv_layer):
                        self.layers[i].differentiateConv(
                            self.layers[i+1].derivatives, self.layers[i+1].oldStencil)

                    if i == 0:
                        self.layers[i].adjustParams(
                            np.reshape(point, self.inputDim), self.eta)
                    else:
                        self.layers[i].adjustParams(
                            self.layers[i-1].values, self.eta)

            iters += 1
            print('iterations: ', iters)

    def __oneHot(self, x):
        vect = np.zeros(self.layers[-1].size)
        vect[int(x)] = 1
        return vect

    def __forwardPass(self, x, y):
        self.layers[0].compute(x)
        for i in range(1, self.layers.size):
            self.layers[i].compute(self.layers[i-1].values)
        return np.argmax(self.layers[-1].values) # predict the maximum value in the output layer

    def __earlyStopping(self, data, iteration): # decides when to stop the training
        if (iteration % 10 != 0): return False
        if iteration == 0: # if it's the first iterati skip everything here right away
            return False
        else: # check only every 10'th iteration
            index = np.random.randint(data.shape[0], size = int(data.shape[0]*0.2)) # create random index set to sample
            sample = data[index,:] # create random sample from the data
            predictions = self.predict(
                np.reshape(sample[:, :-1],
                           (sample.shape[0], self.inputDim[1], self.inputDim[2])),
                sample[:, -1])[0]  # predict on the sample
            accuracy = self.__accuracy(predictions, sample[:, -1])
            print(f'current accuracy: {accuracy}')
            # if accuracy achieved target accuracy, stop training
            if (accuracy >= self.targetAccuracy):
                return True
            return False

    def __accuracy(self, preds, y):
        if (y.size == 0):
            raise Exception('empty sample')
        errors = 0
        for i in range(y.size):
            if (int(preds[i]) != int(y[i])):
                errors += 1
        return 1 - errors/len(y)

    def predict(self, x, y): # public prediction method
        preds = np.empty(x.shape[0], int)
        for i in range(x.shape[0]):
            preds[i] = self.__forwardPass(np.reshape(x[i], self.inputDim), y[i])
        acc = self.__accuracy(preds, y)
        return preds, acc
