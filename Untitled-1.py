from email.mime import image
import numpy as np
import scipy.signal
import skimage
import math
from sklearn.model_selection import train_test_split

import gzip
import sys
import os
import copy
import pickle

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)

def load_mnist_images(filename):
    if not os.path.exists(filename):
        download(filename)
    # Read the inputs in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1,784)
    return data / np.float32(256)

def load_mnist_labels(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
        #data2 = np.zeros( (len(data),10), dtype=np.float32 )
        #for i in range(len(data)):
        #    data2[i][ data[i] ] = 1.0
    return data

train_data = load_mnist_images('train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('train-labels-idx1-ubyte.gz')

train_data = train_data[:2000]
train_labels = train_labels[:2000]

train_data = train_data.reshape((60000, 28, 28))

class ConvLayer(): # convolutional layer
    def __init__(self, gridSize = (3,3), filterCount = 1):
        '''
        Args:
            gridSize(tuple of int): dictates dimension of stencil
            stride(int): how many pixels the stencil is moved after computing a dot product
        '''
        assert (type(gridSize) == tuple) & (len(gridSize) == 2), "Invalid argument for gridSize. Give as a two-tuple in the form: gridSize = (x,y)."
        self.values = None
        self.filterCount = filterCount
        self.layerType = "conv"
        self.stencil = np.random.rand(filterCount, gridSize[0], gridSize[1]) - 0.5 # initialize stencil to random values
        self.oldStencil = None
        self.derivatives = None
        self.outputDim = None
        self.b = np.random.rand(filterCount, gridSize[0], gridSize[1]) - 0.5

    def construct(self, inputDim): # set output dimension
        self.outputDim = (self.filterCount, inputDim[0]-self.gridSize[0]+1, inputDim[1]-self.gridSize[1]+1)
    
    def compute(self, data):
        assert (data.shape[0] >= self.stencil.shape[0]) & (data.shape[1] >= self.stencil.shape[1]), "Stencil larger than input data. Choose smaller stencil or larger data vectors."
        self.values = np.empty(self.outputDim)
        for i in range(data.shape[0]):
            self.values[i] = scipy.signal.correlate(data, self.stencil[i], mode = "valid") + self.b[i] # convolute i'th filter over data

        self.values[self.values < 0] = 0 # apply ReLU
    
    def differentiateDense(self, wNextLayer, nextLayerDerivative): # needs to be called with derivative and w of next layer
        self.derivatives = np.dot(wNextLayer.T, nextLayerDerivative)

    def differentiatePool(self, positions):
        self.derivatives = np.multiply(self.values, positions)

    def differentiateConv(self, nextLayerDerivative, nextLayerStencil):
        self.derivatives = np.empty(self.outputDim)
        for i in range(self.outputDim[0]):
            self.derivatives[i] = scipy.signal.correlate(nextLayerDerivative[i], np.flip(np.flip(nextLayerStencil[i], 0), 1), mode = "full")

    def adjustParams(self, prevLayerVals, eta): # TODO: not sure about derivatives in case of multiple filters
        self.oldStencil = self.stencil
        arr = skimage.util.view_as_windows(prevLayerVals, self.gridSize)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                self.stencil -= arr[i,j] * self.derivatives[i,j] * eta
        self.b -= self.derivatives * eta      

class PoolLayer(): # pooling layer
    def __init__(self, poolSize=3, stride=None):
        assert (type(poolSize) == int), "Invalid argument for pool size. Please give an integer."
        self.poolSize = poolSize
        self.values = None
        self.pos = None # position of the max value per square
        self.layerType = "pool"
        self.outputDim = None
        if (stride == None): self.stride = poolSize
        else: self.stride = stride

    def construct(self, inputDim): # set output dimension
        self.outputDim = (inputDim[0], math.ceil((inputDim[1]-self.poolSize)/self.stride) + 1, math.ceil((inputDim[2]-self.poolSize)/self.stride) + 1)
        self.pos = np.empty(inputDim)
    
    def compute(self, data):
        f = self.poolSize
        a,m,n = data.shape

        for i in range(a):
            # pad the matrix if not evenly divisible by kernel size
            ny = math.ceil(m/self.stride)
            nx = math.ceil(n/self.stride)
            size = ((ny-1)*self.stride+f, (nx-1)*self.stride+f)
            mat_pad = np.full(size, 0)
            mat_pad[:m, :n] = data[i]
            view = self.__asStride(mat_pad, (f, f), self.stride)
            result = np.nanmax(view, axis=(2, 3), keepdims=True)
            pos = np.where(result == view, 1, 0)
            result = np.squeeze(result)
            self.values = result
            self.values[self.values < 0] = 0 # apply ReLU
            self.pos[i] = np.reshape(pos.flatten("K"), (self.outputDim[1]*self.poolSize, self.outputDim[2]*self.poolSize))[:data.shape[1], :data.shape[2]]

    def __asStride(self, arr, sub_shape, stride):
        '''Get a strided sub-matrices view of an ndarray.
        Args:
            arr (ndarray): input array of rank 2 or 3, with shape (m1, n1) or (m1, n1, c).
            sub_shape (tuple): window size: (m2, n2).
            stride (int): stride of windows in both y- and x- dimensions.
        Returns:
            subs (view): strided window view.
        See also skimage.util.shape.view_as_windows()
        '''
        s0, s1 = arr.strides[:2]
        m1, n1 = arr.shape[:2]
        m2, n2 = sub_shape[:2]
        view_shape = (1+(m1-m2)//stride, 1+(n1-n2)//stride, m2, n2)+arr.shape[2:]
        strides = (stride*s0, stride*s1, s0, s1)+arr.strides[2:]
        subs = np.lib.stride_tricks.as_strided(
            arr, view_shape, strides=strides, writeable=False)
        return subs
    
    def differentiateDense(self, wNextLayer, nextLayerDerivative): # needs to be called with derivative and w of next layer
        self.derivatives = np.dot(wNextLayer.T, nextLayerDerivative)

    def differentiateConv(self, nextLayerDerivative, nextLayerStencil):
        self.derivatives = np.empty(self.outputDim)
        for i in range(self.outputDim[0]):
            self.derivatives[i] = scipy.signal.correlate(nextLayerDerivative[i], np.flip(np.flip(nextLayerStencil[i], 0), 1), mode = "full")

    def adjustParams(self, prevLayerVals, eta):
        return

lay = PoolLayer(2)
lay.construct((2,12,12))
arr = np.array([[[1,1,1],[0,0,0],[-1,-1,-1]], [[1,1,1],[0,0,0],[-1,-1,-1]]])
lay.differentiateConv(np.ones((2,4,4)), arr)
lay.derivatives



lay = PoolLayer("max", 2, 2)
arr = np.arange(12).reshape(3,4)
lay.compute(arr)
lay.getPosition()

class DenseLayer(): # layer class
    def __init__(self, size, layerType="hidden"):
        self.w = None # w matrix (j'th row stores a row vector of weights of j'th neuron)
        self.b = None # b vector (b[j] is the bias term of the j'th neuron in this layer)
        self.size = size # number of neurons in this layer
        self.values = None # vector of values
        self.derivatives = None # vector of derivatives
        self.layerType = layerType # hidden, input or output
        self.outputDim = None # dimension of output
    
    def construct(self, w, b, inputDim = None): # initialize random w and b vectors
        self.w = w
        self.b = b
        self.outputDim = tuple([self.size, 1])

    def __softmax(self, x):
        maxi = np.max(x) # normalize the values to prevent overflows
        sumAll = sum(np.exp(x-maxi))
        return np.exp(x-maxi)/sumAll
        
    def compute(self, valPrevLayer):
        if (valPrevLayer.shape[0] != valPrevLayer.size): # flatten the input
            valPrevLayer = valPrevLayer.reshape(valPrevLayer.size, 1)
        self.values = np.dot(self.w, valPrevLayer) + self.b
        if (self.layerType != "output"): self.values[self.values < 0] = 0 # apply ReLU
        else:
            self.values = self.__softmax(self.values)

    def differentiateDense(self, wNextLayer, nextLayerDerivative): # needs to be called with derivative and w of next layer
        self.derivatives = np.dot(wNextLayer.T, nextLayerDerivative)
        
    def adjustParams(self, valPrevLayer, eta):
        self.b -= np.reshape(self.derivatives, (self.size,)) * eta
        self.w -= np.dot(np.reshape(self.derivatives, (self.size, 1)), np.reshape(valPrevLayer.T, (1,valPrevLayer.size)))*eta
        

class FFNN(): # neural network class
    def __init__(self, learning_rate = 0.1, batch_size = 0.1, targetAccuracy = 0.9):
        self.learning_rate = learning_rate
        self.batch_size = batch_size # share of the dataset used to update params
        self.layers = [DenseLayer(None, "input")] # initialize an input layer
        self.targetAccuracy = targetAccuracy # when this accuracy (plus sufficient convergence) is achieved, training is halted\

    def addLayer(self, layer):
        self.layers.append(layer)
        
    def __construct(self, d, k = 1): # private constructor method to create layers and initialize params
        # d is number of features
        # k is number of classes in classification

        self.layers.append(DenseLayer(k, "output")) # add output layer
        self.layers = np.array(self.layers) # convert layer list to Numpy array

        #initialize input layer
        self.layers[0].size = d
        self.layers[0].construct(None, np.random.rand(d) - 0.5) # initialize to random b vector

        for h in range(1, len(self.layers)): # create hidden layers
            self.layers[h].construct(np.random.rand(self.layers[h].size, self.layers[h-1].size) - 0.5, np.random.rand(self.layers[h].size) - 0.5)
    
    def train(self, x, y):
        iters = 0
        eta = self.learning_rate
        self.__construct(x.shape[1], len(np.unique(y))) # call internal constructor method
        
        data = np.concatenate((x, y), 1) # merge x and y data such that the data/labels correspond
        batchSize = int(self.batch_size * x.shape[0])
        
        while(not self.__earlyStopping(x,y,iters)):
            # randomly select batch to train on
            index = np.random.randint(x.shape[0], size = batchSize) # create random index set to sample
            batch = data[index,:] # create random sample from the data
            
            for sample in range(batchSize): # iterate through all samples in the batch
                point = batch[sample, :-1]
                label = batch[sample, -1]
                self.__forwardPass(point, label) # call internal prediction method
                
                # compute derivative with respect to nodes in last layer
                self.layers[-1].derivatives = (self.layers[-1].values - self.__oneHot(label))/batchSize
                self.layers[-1].adjustParams(self.layers[-2].values, eta)
    
                for i in range(len(self.layers)-2, 0, -1):
                    self.layers[i].differentiateDense(self.layers[i+1].w, self.layers[i+1].derivatives.reshape(self.layers[i+1].size, 1))
                    self.layers[i].adjustParams(self.layers[i-1].values, eta)
                        
            iters += 1
            print("iterations: ", iters)
                    
    def __oneHot(self, x):
        vect = np.zeros(self.layers[-1].size)
        vect[int(x)] = 1
        return vect
    
    def __forwardPass(self, x, y): # accepts x vector and y label (forward pass for one label)
        self.layers[0].values = x
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
            
    def __earlyStopping(self, x, y, iteration): # decides when to stop the training
        global oldAccuracy
        if (iteration % 10 != 0): return False 
        if iteration == 0: # if it's the first iteration, skip everything here right away
            oldAccuracy = 0.3 # set accuracy to 30%
            return False
        else: # check only every 10'th iteration
            data = np.concatenate((x, y), 1) # merge x and y so that they correspond (to sample from it)
            index = np.random.randint(x.shape[0], size = int(x.shape[0]*0.2)) # create random index set to sample
            sample = data[index,:] # create random sample from the data
            predictions = self.predict(sample[:,:-1], sample[:,-1])[0] # predict on the sample
            accuracy = self.__accuracy(predictions, sample[:, -1])
            print("current accuracy: ", accuracy)
            
            # if accuracy barely changes and achieved target accuracy, stop training
            if ((abs(accuracy - oldAccuracy) < 0.01) & (accuracy > self.targetAccuracy)): return True
            oldAccuracy = accuracy
            return False
    
    def __accuracy(self, preds, y):
        if (y.size == 0): raise Exception("empty sample")
        errors = 0
        for i in range (y.size):
            if (preds[i] != int(y[i])): errors += 1
        return 1 - errors/y.size

trainX, testX, trainY, testY = train_test_split(train_data, train_labels, test_size = 0.3)

mynn = FFNN(batch_size = 0.1, learning_rate = 0.04)
mynn.addLayer(DenseLayer(80, "hidden"))
mynn.addLayer(DenseLayer(50, "hidden"))
mynn.train(trainX, trainY.reshape(trainY.shape[0], 1))
mynn.predict(testX, testY)

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
        self.layers.append(DenseLayer(k, "output")) # add output layer
        self.layers = np.array(self.layers)
        self.layers[-1].construct(None, np.random.rand(k) - 0.5)

        assert (type(self.layers[0]) != DenseLayer), "First layer cannot be dense."
        self.layers[0].construct(x.shape)

        for l in range(1, self.layers):
            if (self.layers[l].layerType == "hidden"): 
                self.layers[l].construct(np.random.rand(self.layers[l].size, self.layers[l-1].outputSize[0]*self.layers[l-1].outputSize[1])-0.5, np.random.rand(self.layers[l].size)-0.5)
            else: self.layers[l].construct(self.layers[l-1].outputDim)


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
        newInput = np.empty((len(x), layers[-1].getSize()))
        for i in range(len(x)): # iterate through samples
            layers[0].compute(x[i])
            for j in range(layers.size -1): # propagate the value through all layers
                layers[j+1].compute(layers[j].getValues())
            newInput[i] = self.__vectorize(self.layers[-1].getValues()) # store value as input for ffnn
            
        return self.ffnn.predict(newInput, y)

net = CNN(targetAccuracy=0.8)
net.addLayer(ConvLayer(filterCount=2))
net.addLayer(PoolLayer())
net.train(trainX, trainY)
net.predict(testX, testY)

lay = ConvLayer((2,2), 1)
arr = np.arange(16).reshape(4,4)
filt = np.ones((2,2))
scipy.signal.correlate(arr, filt, mode = "same")
lay = PoolLayer()
arr = np.arange(25).reshape(5,5)
lay.construct((5,5))
lay.compute(arr)
posers = lay.pos

np.multiply(posers, np.arange(25).reshape(5,5))
