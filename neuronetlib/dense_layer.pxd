import numpy as np
cimport numpy as np
from layer cimport layer

cdef class dense_layer(layer):
    cdef np.ndarray w
    cdef np.ndarray b
    cdef int size
    cdef str layerType

    cdef void construct(self, tuple inputDim) # initialize random w and b vectors
    cdef np.ndarray __softmax(self, np.ndarray x)
    cdef void compute(self, np.ndarray valPrevLayer)
    cdef void differentiateDense(self, np.ndarray wNextLayer,
                                 np.ndarray nextLayerDerivative) # needs to be called with derivative and w of next layer
    cdef void differentiateConv(self, np.ndarray nextLayerDerivative,
                                np.ndarray nextLayerStencil)
    cdef void differentiatePool(self, np.ndarray positions,
                                np.ndarray valPrevLayer)
    cdef void adjustParams(self, np.ndarray valPrevLayer, double eta)
