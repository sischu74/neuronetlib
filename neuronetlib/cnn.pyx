# distutils: language = c++
cimport numpy as np
import numpy as np
from layer cimport layer
from dense_layer cimport dense_layer
from conv_layer cimport conv_layer
from pool_layer cimport pool_layer
# from layer import layer
# from dense_layer import dense_layer
# from conv_layer import conv_layer
# from pool_layer import pool_layer

cdef class CNN(): # convolutional neural network class
    cdef list Layers
    cdef np.ndarray layers
    cdef double eta
    cdef double batch_size
    cdef double targetAccuracy
    cdef list inputDim

    def __init__(self, double learning_rate = 0.1,
                  double batch_size = 0.1, double targetAccuracy = 0.9):
        self.Layers = []
        self.eta = learning_rate
        self.batch_size = batch_size
        self.targetAccuracy = targetAccuracy
        self.inputDim = None

    def addLayer(self, layer):
        self.Layers.append(layer)

    cdef void __construct(self, np.ndarray x, int k):
        # x is a sample input (used for input size), k is number of output classes
        cdef layer curr_layer, prev_layer
        self.inputDim = list([1, x.shape[0], x.shape[1]])
        self.Layers.append(dense_layer(k, 'output')) # add output layer
        self.layers = np.asarray(self.Layers)

        # if isinstance(self.layers[0], dense_layer):
        #     curr_layer = self.layers[0]
        #     curr_layer.construct(np.random.rand(self.layers[0].size, x.size)-0.5,
        #                              np.random.rand(self.layers[0].size)-0.5)
        # else:
        curr_layer = self.layers[0]
        curr_layer.construct((1, x.shape[0], x.shape[1]))

        # cdef bint first = True # indicates if current layer is the first dense hidden layer
        cdef int l
        # cdef tuple predDim
        for l in range(1, len(self.layers)):
            # if isinstance(self.layers[l], dense_layer):
            #     if first and isinstance(self.layers[l-1], dense_layer):
            #         first = False
            #         predDim = self.layers[l-1].outputDim # dimensions of last non-dense layer
            #         self.layers[l].construct(predDim)
            #     else:
            #         # assert(isinstance(self.layers[l-1], dense_layer)), 'Layer after dense layer must be dense too.'
            #         self.layers[l].construct(predDim)
            # else:
            curr_layer = self.layers[l]
            prev_layer = self.layers[l-1]
            curr_layer.construct(prev_layer.output_dim)


    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        self.ctrain(x, y)

    cdef void ctrain(self, np.ndarray x, np.ndarray y):
        cdef np.ndarray data
        cdef np.ndarray index
        cdef np.ndarray batch
        cdef np.ndarray point
        cdef int label, sample, batchSize, i, iters = 0
        cdef layer curr_layer

        self.__construct(x[0], len(np.unique(y)))

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
                curr_layer = self.layers[-1]
                prev_layer = self.layers[-2]
                curr_layer.derivatives = (
                    curr_layer.values - self.__oneHot(label))/batchSize
                curr_layer.adjustParams(prev_layer.values, self.eta)

                for i in range(len(self.layers)-2, -1, -1):
                    curr_layer = self.layers[i+1]
                    prev_layer = self.layers[i]
                    if isinstance(curr_layer, dense_layer):
                        prev_layer.differentiateDense(
                            curr_layer.w, curr_layer.derivatives.reshape(
                                curr_layer.size, 1))
                    elif isinstance(curr_layer, pool_layer):
                        prev_layer.differentiatePool(
                            curr_layer.fullDerMat, curr_layer.pos)
                    elif isinstance(curr_layer, conv_layer):
                        prev_layer.differentiateConv(
                            curr_layer.derivatives, curr_layer.oldStencil)

                    if i == 0:
                        prev_layer.adjustParams(
                            np.reshape(point, self.inputDim), self.eta)
                    else:
                        curr_layer = self.layers[i-1]
                        prev_layer.adjustParams(
                            curr_layer.values, self.eta)

            iters += 1
            print(f'iterations: {iters}')

    cdef np.ndarray __oneHot(self, int x):
        cdef dense_layer last_layer = self.layers[-1]
        cdef np.ndarray vect
        vect = np.zeros(last_layer.size)
        vect[x] = 1
        return vect

    cdef int __forwardPass(self, np.ndarray x, int y):
        cdef int i
        cdef layer curr_layer = self.layers[0]
        cdef layer prev_layer
        curr_layer.compute(x)
        for i in range(1, self.layers.size):
            curr_layer = self.layers[i]
            prev_layer = self.layers[i-1]
            curr_layer.compute(prev_layer.values)
        curr_layer = self.layers[-1]
        return np.argmax(curr_layer.values) # predict the maximum value in the output layer

    cdef bint __earlyStopping(self, np.ndarray data, int iteration): # decides when to stop the training
        cdef np.ndarray index
        cdef np.ndarray sample
        cdef np.ndarray predictions
        cdef double accuracy
        if (iteration % 10 != 0): return False
        if iteration == 0: # if it's the first iterati skip everything here right away
            return False
        else: # check only every 10'th iteration
            index = np.random.randint(data.shape[0], size = int(data.shape[0]*0.2)) # create random index set to sample
            sample = data[index,:] # create random sample from the data
            predictions = self.predict(
                np.reshape(sample[:, :-1],
                           (sample.shape[0], self.inputDim[1], self.inputDim[2])),
                sample[:, -1])  # predict on the sample
            accuracy = self.__accuracy(predictions, sample[:, -1])
            print(f'current accuracy: {accuracy}')
            # if accuracy achieved target accuracy, stop training
            if (accuracy >= self.targetAccuracy):
                return True
            return False

    cdef double __accuracy(self, np.ndarray preds, np.ndarray y):
        cdef int errors
        cdef int i
        if (y.size == 0):
            raise Exception('empty sample')
        errors = 0
        for i in range(y.size):
            if (int(preds[i]) != int(y[i])):
                errors += 1
        return 1 - errors/len(y)

    cdef np.ndarray predict(self, np.ndarray x, np.ndarray y): # public prediction method
        cdef np.ndarray preds
        cdef int i
        preds = np.empty(x.shape[0], int)
        for i in range(x.shape[0]):
            preds[i] = self.__forwardPass(np.reshape(x[i], self.inputDim), y[i])
        return preds
