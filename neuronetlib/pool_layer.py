#!/usr/bin/env python3
import numpy as np
import math

class Pool_layer(): # pooling layer

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
