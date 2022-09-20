#!/usr/bin/env python3
import numpy as np
import scipy.signal
import skimage
from layer import Layer


class Conv_layer(Layer): # convolutional layer

    def __init__(self, grid_size = (3,3), filterCount = 1):
        '''
        Args:
            grid_size(tuple of int): dictates dimension of stencil
            stride(int): how many pixels the stencil is moved after computing a dot product
        '''
        assert (type(grid_size) == tuple) & (len(grid_size) == 2), "Invalid argument for grid_size. Give as a two-tuple in the form: grid_size = (x,y)."
        self.values = None
        self.grid_size = grid_size
        self.filterCount = filterCount
        self.stencil = np.random.rand(filterCount, grid_size[0], grid_size[1]) - 0.5 # initialize stencil to random values
        self.oldStencil = None
        self.derivatives = None
        self.outputDim = None
        self.b = np.random.rand(filterCount, grid_size[0], grid_size[1]) - 0.5

    def construct(self, inputDim: tuple) -> None:  # set output dimension
        self.outputDim = (self.filterCount, inputDim[0]-self.grid_size[0]+1, inputDim[1]-self.grid_size[1]+1)

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
        arr = skimage.util.view_as_windows(prevLayerVals, self.grid_size)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                self.stencil -= arr[i,j] * self.derivatives[i,j] * eta
        self.b -= self.derivatives * eta
