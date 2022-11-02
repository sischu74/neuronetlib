cimport numpy as np
import numpy as np
from layer cimport layer

cdef class conv_layer(layer): # convolutional layer
    cdef tuple grid_size
    cdef int filter_count
    cdef np.ndarray stencil
    cdef np.ndarray oldStencil
    cdef np.ndarray inputDim
    cdef np.ndarray b

    cdef void construct(self, tuple inputDim) # set output dimension
    cdef void compute(self, np.ndarray data)
    cdef void differentiateDense(self, np.ndarray wNextLayer,
                                 np.ndarray nextLayerDerivative) # needs to be called with derivative and w of next layer
    cdef void differentiatePool(self, np.ndarray nextLayerDerivative,
                                np.ndarray positions)
    cdef void differentiateConv(self, np.ndarray nextLayerDerivative,
                                np.ndarray nextLayerStencil)
    cdef void adjustParams(self, np.ndarray prevLayerVals, double eta)
