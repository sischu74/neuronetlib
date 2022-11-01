# distutils: language = c++
import numpy as np
cimport numpy as np
from layer cimport layer
from layer import layer


cdef class layer:
    """Abstract base class for different layers."""
    def __init__(self):
        pass

    cdef void construct(self, tuple inputDim):
        pass

    cdef void compute(self, np.ndarray data):
        pass

    cdef void differentiateDense(self, np.ndarray wNextLayer,
                           np.ndarray nextLayerDerivative):
        pass

    cdef void differentiateConv(self, np.ndarray nextLayerDerivative,
                                np.ndarray nextLayerStencil):
        pass

    cdef void differentiatePool(self, np.ndarray positions,
                                np.ndarray nextLayerDerivative):
        pass

    cdef void adjustParams(self, np.ndarray valPrevLayer, double eta):
        pass
