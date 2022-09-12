#!/usr/bin/env python3
import numpy as np
import scipy
import skimage
import math

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
