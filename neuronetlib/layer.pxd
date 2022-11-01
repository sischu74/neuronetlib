cimport numpy as np
import numpy as np

cdef class layer:
    """Abstract base class for different layers."""
    cdef tuple output_dim
    cdef np.ndarray values
    cdef np.ndarray derivatives

    cdef void construct(self, tuple inputDim)
    cdef void compute(self, np.ndarray data)
    cdef void differentiateDense(self, np.ndarray wNextLayer,
                           np.ndarray nextLayerDerivative)
    cdef void differentiateConv(self, np.ndarray nextLayerDerivative,
                                np.ndarray nextLayerStencil)
    cdef void differentiatePool(self, np.ndarray positions,
                                np.ndarray nextLayerDerivative)
    cdef void adjustParams(self, np.ndarray prevLayerVals, double eta)
