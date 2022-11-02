# distutils: language = c++
import numpy as np
cimport numpy as np
import scipy.signal
from layer cimport layer


cdef class conv_layer(layer): # convolutional layer
    def __init__(self, tuple grid_size = (3,3), int filter_count = 1):
        '''
        Args:
            grid_size(tuple of int): dictates dimension of stencil
            stride(int): how many pixels the stencil is moved after computing a dot product
        '''
        assert (type(grid_size) == tuple) & (len(grid_size) == 2), "Invalid argument for grid_size. Give as a two-tuple in the form: grid_size = (x,y)"
        self.grid_size = grid_size
        self.filter_count = filter_count
        self.stencil = np.random.rand(filter_count, grid_size[0],
                                      grid_size[1]) - 0.5 # initialize stencil to random values
        self.oldStencil = None
        self.derivatives = None
        self.values = None
        self.inputDim = None
        self.output_dim = None

    cdef void construct(self, tuple inputDim): # set output dimension
        self.inputDim = np.array(inputDim)
        self.output_dim = (self.filter_count * inputDim[0],
                          inputDim[1]-self.grid_size[0]+1,
                                     inputDim[2]-self.grid_size[1]+1)
        self.values = np.zeros(self.output_dim)

    cdef void compute(self, np.ndarray data):
        cdef int i
        for i in range(self.filter_count):
            self.values[i:i+self.inputDim[0]] = scipy.signal.correlate(
                data, np.array([self.stencil[i]]), mode='valid')
        self.values[self.values < 0] = 0 # apply ReLU

    cdef void differentiateDense(self, np.ndarray wNextLayer,
                                 np.ndarray nextLayerDerivative): # needs to be called with derivative and w of next layer
        self.derivatives = np.reshape(np.dot(wNextLayer.T, nextLayerDerivative),
                                      self.output_dim)

    cdef void differentiatePool(self, np.ndarray nextLayerDerivative,
                                np.ndarray positions):
        self.derivatives = np.multiply(nextLayerDerivative, positions)

    cdef void differentiateConv(self, np.ndarray nextLayerDerivative,
                                np.ndarray nextLayerStencil):
        self.derivatives = np.empty(self.output_dim)
        self.derivatives = scipy.signal.correlate(nextLayerDerivative, np.flip(
            np.flip(nextLayerStencil, 0), 1), mode = 'full')

    cdef void adjustParams(self, np.ndarray prevLayerVals, double eta):
        cdef int i
        cdef int l
        self.oldStencil = self.stencil
        for i in range(self.filter_count):
            for l in range(self.inputDim[0]):
                self.stencil[i] -= eta * scipy.signal.correlate(
                    prevLayerVals[l],
                    self.derivatives[i*self.inputDim[0] + l], mode='valid')
