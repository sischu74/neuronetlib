#!/usr/bin/env python3

import gzip
import sys
import os
import numpy as np
if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve


class Initializer():
    """
    Initialize and load the mnist dataset.
    """

    def __init__(self, file_name):
        self.file_name = file_name

    def initalize(self) -> np.array:
        train_data = self.load_mnist_images('train-images-idx3-ubyte.gz')
        train_labels = self.load_mnist_labels('train-labels-idx1-ubyte.gz')
        return train_data, train_labels

    def download(self, filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    def load_mnist_images(self, filename):
        if not os.path.exists(filename):
            self.download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1,784)
        return data / np.float32(256)

    def load_mnist_labels(self, filename):
        if not os.path.exists(filename):
            self.download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data
