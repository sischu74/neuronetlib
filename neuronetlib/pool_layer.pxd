cimport numpy as np
from layer cimport layer

cdef class pool_layer(layer): # pooling layer
    cdef int pool_size
    cdef np.ndarray pos
    cdef int size
    cdef np.ndarray fullDerMat
    cdef tuple inputDim
    cdef int stride

    cdef void construct(self, tuple inputDim)  # set output dimension
    cdef void compute(self, np.ndarray data)
    cdef np.ndarray __asStride(self, arr,
                               tuple sub_shape, int stride)
    cdef void differentiateDense(self, np.ndarray wNextLayer,
                                 np.ndarray nextLayerDerivative) # needs to be called with derivative and w of next layer
    cdef void differentiateConv(self, np.ndarray nextLayerDerivative,
                                np.ndarray nextLayerStencil)
    cdef void differentiatePool(self, np.ndarray positions,
                                np.ndarray nextLayerDerivative)
    cdef void adjustParams(self, np.ndarray prevLayerVals, double eta)
