#!/usr/bin/env python3
import init
from cnn import CNN
from dense_layer import Dense_layer
from conv_layer import Conv_layer
from pool_layer import Pool_layer
from sklearn.model_selection import train_test_split

train_data, train_labels = init.initalize()
train_data = train_data.reshape((train_data.shape[0], 28, 28))

trainX, testX, trainY, testY = train_test_split(train_data, train_labels,
                                                test_size=0.3)

cnn = CNN(targetAccuracy=0.8)
cnn.addLayer(Conv_layer(filterCount=2))
cnn.addLayer(Pool_layer())
cnn.train(trainX, trainY)
