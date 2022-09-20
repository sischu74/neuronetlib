from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    """
    Abstract base class for different layers.
    """

    def __init__(self, args):
        super(Layer, self).__init__()

    @abstractmethod
    def construct(self, inputDim: tuple) -> None:
        pass

    @abstractmethod
    def compute(self, data) -> None:
        pass

    @abstractmethod
    def differentiateDense(self, wNextLayer, nextLayerDerivative) -> None:
        pass

    @abstractmethod
    def differentiateConv(self, nextLayerDerivative, nextLayerStencil) -> None:
        pass

    @abstractmethod
    def differentiatePool(self, positions) -> None:
        pass

    @abstractmethod
    def adjustParams(self, prevLayerVals: np.ndarray, eta: float) -> None:
        pass
