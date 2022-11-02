#!/usr/bin/env python3
import numpy as np
cimport numpy as np
from layer cimport layer

cdef class dense_layer(layer): # layer class
    def __init__(self, int size, str layerType='hidden'):
        self.w = None # w matrix (j'th row stores a row vector of weights of j'th neuron)
        self.b = None # b vector (b[j] is the bias term of the j'th neuron in this layer)
        self.size = size # number of neurons in this layer
        self.output_dim = None
        self.values = None # vector of values
        self.derivatives = None # vector of derivatives
        self.layerType = layerType # hidd input or output

    cdef void construct(self, tuple inputDim): # initialize random w and b vectors
        self.w = np.random.rand(self.size,
                                inputDim[0]*inputDim[1]*inputDim[2]) - 0.5
        self.b = np.random.rand(self.size) - 0.5

        self.output_dim = (self.size, 1, 1)

    cdef np.ndarray __softmax(self, np.ndarray x):
        cdef double maxi
        cdef double sumAll
        maxi = np.max(x) # normalize the values to prevent overflows
        sumAll = sum(np.exp(x-maxi))
        return np.exp(x-maxi)/sumAll

    cdef void compute(self, np.ndarray valPrevLayer):
        # if self.is_multi_dimensional(valPrevLayer) > 1:  # if input comes from a non-dense layer, flatten it
        valPrevLayer = valPrevLayer.reshape(valPrevLayer.size)
        self.values = np.dot(self.w, valPrevLayer) + self.b
        if (self.layerType != 'output'):
            self.values[self.values < 0] = 0 # apply ReLU
        else:
            self.values = self.__softmax(self.values)

    cdef void differentiateDense(self, np.ndarray wNextLayer,
                                 np.ndarray nextLayerDerivative): # needs to be called with derivative and w of next layer
        self.derivatives = np.dot(wNextLayer.T, nextLayerDerivative)

    cdef void differentiateConv(self, np.ndarray nextLayerDerivative,
                                np.ndarray nextLayerStencil):
        raise Exception("Not supported: Convolutional layer after dense layer")

    cdef void differentiatePool(self, np.ndarray positions,
                                np.ndarray valPrevLayer):
        raise Exception("Not supported: Pooling layer after dense layer")

    cdef void adjustParams(self, np.ndarray valPrevLayer, double eta):
        # if self.is_multi_dimensional(valPrevLayer): # if input comes from a non-dense layer, flatten it
        valPrevLayer = valPrevLayer.reshape(valPrevLayer.size)
        if self.layerType == 'input':
            return
        self.b -= self.derivatives.reshape(self.derivatives.size,) * eta
        self.w -= np.dot(np.reshape(self.derivatives, (self.size, 1)),
                         np.reshape(valPrevLayer.T, (1, valPrevLayer.size)))*eta

# class Dense_layer(Layer):
#     """
#     Dense Layer class implements abstract Layer class. Simple propagating layer.
#     """

#     def __init__(self, size: int, layer_type: str = "hidden"):
#         self.w = None # w matrix (j'th row stores a row vector of weights of j'th neuron)
#         self.b = None # b vector (b[j] is the bias term of the j'th neuron in this layer)
#         self.size = size # number of neurons in this layer
#         self.layer_type = layer_type # output or hidden
#         self.values = None # vector of values
#         self.derivatives = None # vector of derivatives
#         self.outputDim = None # dimension of output

#     def construct(self, w, b, inputDim = None): # initialize random w and b vectors
#         self.w = w
#         self.b = b
#         self.outputDim = tuple([self.size, 1])

#     def __softmax(self, x):
#         maxi = np.max(x) # normalize the values to prevent overflows
#         sumAll = sum(np.exp(x-maxi))
#         return np.exp(x-maxi)/sumAll

#     def compute(self, valPrevLayer):
#         if (valPrevLayer.shape[0] != valPrevLayer.size): # flatten the input
#             valPrevLayer = valPrevLayer.reshape(valPrevLayer.size, 1)
#         self.values = np.dot(self.w, valPrevLayer) + self.b
#         if (self.layerType != "output"):
#             self.values[self.values < 0] = 0 # apply ReLU
#         else:
#             self.values = self.__softmax(self.values)

#     def differentiateDense(self, wNextLayer, nextLayerDerivative): # needs to be called with derivative and w of next layer
#         self.derivatives = np.dot(wNextLayer.T, nextLayerDerivative)


#     def adjustParams(self, valPrevLayer, eta):
#         self.b -= np.reshape(self.derivatives, (self.size,)) * eta
#         self.w -= np.dot(np.reshape(self.derivatives, (self.size, 1)), np.reshape(valPrevLayer.T, (1,valPrevLayer.size)))*eta



