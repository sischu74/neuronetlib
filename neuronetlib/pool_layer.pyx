# distutils: language = c++
import numpy as np
cimport numpy as np
import math
import scipy.signal
from layer cimport layer
from pool_layer cimport pool_layer

cdef class pool_layer(layer): # pooling layer
    def __init__(self, pool_size=2, stride=0):
        self.pool_size = pool_size
        self.values = None
        self.pos = None  # position of the max value per square
        self.derivatives = None
        self.fullDerMat = None
        self.output_dim = None
        self.inputDim = None
        if stride == 0:
            self.stride = pool_size
        else:
            self.stride = stride

    cdef void construct(self, tuple inputDim):  # set output dimension
        self.output_dim = (inputDim[0],
                          math.ceil((inputDim[1]-self.pool_size)/self.stride) + 1,
                          math.ceil((inputDim[2]-self.pool_size)/self.stride) + 1)
        self.pos = np.empty(inputDim)
        self.inputDim = inputDim
        self.size = self.output_dim[0]*self.output_dim[1]*self.output_dim[2]
        self.values = np.zeros(self.output_dim)

    cdef void compute(self, np.ndarray data):
        cdef int f, a, i, ny, nx
        cdef double m, n
        cdef np.ndarray mat_pad
        cdef np.ndarray view
        cdef np.ndarray result
        cdef np.ndarray pos


        f = self.pool_size
        a = data.shape[0]
        m = data.shape[1]
        n = data.shape[2]

        for i in range(a):
            # pad the matrix if not evenly divisible by kernel size
            ny = math.ceil(m/self.stride)
            nx = math.ceil(n/self.stride)
            size = ((ny-1)*self.stride+f, (nx-1)*self.stride+f)
            mat_pad = np.full(size, 0, dtype=float)
            mat_pad[:int(m), :int(n)] = data[i]
            view = self.__asStride(mat_pad, (f, f), self.stride)
            result = np.nanmax(view, axis=(2, 3), keepdims=True)
            pos = np.where(result == view, 1, 0)
            result = np.squeeze(result)
            self.values[i] = result
            self.pos[i] = np.reshape(pos.flatten('K'),
                        (self.output_dim[1]*self.pool_size,
                         self.output_dim[2]*self.pool_size))[:data.shape[1],
                                                            :data.shape[2]]

    cdef np.ndarray __asStride(self, arr,
                               tuple sub_shape, int stride):
        cdef int s0, s1, m1, n1, m2, n2
        cdef tuple view_shape
        cdef tuple strides
        cdef np.ndarray subs
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

    cdef void differentiateDense(self, np.ndarray wNextLayer,
                                 np.ndarray nextLayerDerivative): # needs to be called with derivative and w of next layer
        self.derivatives = np.reshape(np.dot(wNextLayer.T, nextLayerDerivative),
                                      self.output_dim)
        self.fullDerMat = np.kron(self.derivatives,
                          np.ones((self.stride, self.stride)))[:self.inputDim[0],
                                        :self.inputDim[1], :self.inputDim[2]]

    cdef void differentiateConv(self, np.ndarray nextLayerDerivative,
                                np.ndarray nextLayerStencil):
        self.derivatives = np.empty(self.output_dim)
        self.derivatives = scipy.signal.correlate(nextLayerDerivative,
                                         np.flip(np.flip(nextLayerStencil, 0), 1),
                                         mode='full')

        self.fullDerMat = np.kron(self.derivatives, np.ones((
            self.stride, self.stride)))[:self.inputDim[0],
                                        :self.inputDim[1], :self.inputDim[2]]

    cdef void differentiatePool(self, np.ndarray positions,
                                np.ndarray nextLayerDerivative):
        raise Exception("Not supported: Pooling layer after pooling layer")

    cdef void adjustParams(self, np.ndarray prevLayerVals, double eta):
        return
