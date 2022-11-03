## Deep Learning Library from Scratch
In order to fully understand how artificial neural networks are trained, I wanted to write a Python libray for neural networks from scratch, that is, using only NumPy. Over the course of writing the libray, I started studying CS and rewrote the source code multiple times, adding new features like CNNs and in the most recent iteration rewriting everything in Cython.

The library will be expanded in the future to accomodate more features and improve numerical stability, which is not great at the moment. For too many layers and nodes, overflows occur quickly. Recommended for CNNs is one convolutional and one pooling layer. A dense layer is inserted automatically as the output layer.

The point of the library is for people interested in the lower-level workings of deep learning and libraries in particular to take a look at concise source code and get an idea of how things can be implemented efficiently.

## Using neuronetlib
```Python
# install neuronetlib. Recommended to run in the terminal, since Jupyter Notebooks have difficulties with pip installing libraries.
pip install neuronetlib
```


Import cnn and layer packages.
```Python
import neuronetlib as nnl
from neuronetlib import cnn
from neuronetlib import conv_layer
from neuronetlib import pool_layer
from neuronetlib import dense_layer

# initialize network
network = nnl.cnn.CNN(learning_rate=0.08, batch_size=0.05, targetAccuracy=0.91)

# add a convolutional layer with 4 filters
cnn.addLayer(neuronetlib.conv_layer.conv_layer(grid_size=(3,3), filter_count=4))

# add a pooling layer
cnn.addLayer(neuronetlib.pool_layer.pool_layer(pool_size=2, stride=0))

# add a dense layer
cnn.addLayer(neuronetlib.dense_layer.dense_layer(size=50, layerType='hidden'))

# train network on hypothetical training data
cnn.train(trainX, trainY)

# predict on hypothetical test data
cnn.predict(testX, testY)
```
