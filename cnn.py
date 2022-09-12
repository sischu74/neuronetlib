#!/usr/bin/env python3
import numpy as np
from conv_layer import Conv_layer
from pool_layer import Pool_layer

class CNN(): # convolutional neural network class
    def __init__(self, learning_rate = 0.1, batchSize = 0.1, targetAccuracy = 0.9):
        self.layers = []
        self.eta = learning_rate
        self.batchSize = batchSize
        self.targetAccuracy = targetAccuracy

    def addLayer(self, layer):
        self.layers.append(layer)

    def __construct(self, x, k):
        # x is a sample input (used for input size), k is number of output classes
        self.layers.append(Dense_layer(k, "output")) # add output layer
        self.layers = np.array(self.layers)
        self.layers[-1].construct(None, np.random.rand(k) - 0.5)

        assert (type(self.layers[0]) != Dense_layer), "First layer cannot be dense."
        self.layers[0].construct(x.shape)

        for l in range(1, self.layers):
            if (self.layers[l].layerType == "hidden"):
                self.layers[l].construct(np.random.rand(self.layers[l].size, self.layers[l-1].outputSize[0]*self.layers[l-1].outputSize[1])-0.5, np.random.rand(self.layers[l].size)-0.5)
            else:
                self.layers[l].construct(self.layers[l-1].outputDim)

    def train(self, x, y):
        inputDim = x[0].shape
        self.__construct(x[0], np.unique(y))
        iters = 0

        data = np.reshape(x, (x[0].size,1)) # flatten array
        data = np.concatenate((data, y), 1) # merge x and y data such that the data/labels correspond
        batchSize = int(self.batch_size * x.shape[0])

        while(not self.__earlyStopping(x,y,iters)):
            # randomly select batch to train on
            index = np.random.randint(x.shape[0], size = batchSize) # create random index set to sample
            batch = data[index,:] # create random sample from the data

            for sample in range(batchSize): # iterate through all samples in the batch
                point = batch[sample, :-1]
                label = batch[sample, -1]
                self.__forwardPass(np.reshape(point, inputDim), label) # call internal prediction method

                # compute derivative with respect to nodes in last layer
                self.layers[-1].derivatives = (self.layers[-1].values - self.__oneHot(label))/batchSize
                self.layers[-1].adjustParams(self.layers[-2].values, self.eta)

                for i in range(len(self.layers)-2, 0, -1):
                    currLayer = self.layers[i]
                    if(self.layers[i+1].layerType == "dense"):
                        currLayer.differentiateDense(self.layers[i+1].w, self.layers[i+1].derivatives.reshape(self.layers[i+1].size, 1))
                    elif(self.layers[i+1].layerType == "pool"):
                        currLayer.differentiatePool(self.layers[i+1].pos)
                    elif(self.layers[i+1].layerType == "conv"):
                        currLayer.differentiateConv(self.layers[i+1].derivatives, self.layers[i+1].oldStencil)

                    currLayer.adjustParams(self.layers[i-1].values, self.eta)

            iters += 1
            print("iterations: ", iters)

    def __forwardPass(self, x, y):
        self.layers[0].compute(x)
        for i in range(1, self.layers):
            self.layers[i].compute(self.layers[i-1].values)
        return np.argmax(self.layers[-1].values) # predict the maximum value in the output layer

    def predict(self, x, y):
        newInput = np.empty((len(x), self.layers[-1].getSize()))
        for i in range(len(x)): # iterate through samples
            self.layers[0].compute(x[i])
            for j in range(self.layers.size -1): # propagate the value through all layers
                self.layers[j+1].compute(self.layers[j].getValues())
            newInput[i] = self.__vectorize(self.layers[-1].getValues()) # store value as input for ffnn

        return self.ffnn.predict(newInput, y)
