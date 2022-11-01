import neuronetlib.init
from sklearn.model_selection import train_test_split
import neuronetlib.conv_layer
import neuronetlib.pool_layer
import neuronetlib.cnn

if __name__ == "__main__":
    train_data, train_labels = neuronetlib.init.initalize()
    train_data = train_data.reshape((train_data.shape[0], 28, 28))

    trainX, testX, trainY, testY = train_test_split(train_data, train_labels,
                                                    test_size=0.3)

    cnn = neuronetlib.cnn.CNN(batch_size=0.05, targetAccuracy=0.92)
    cnn.addLayer(neuronetlib.conv_layer.conv_layer(filter_count=4))
    cnn.addLayer(neuronetlib.pool_layer.pool_layer())
    cnn.addLayer(neuronetlib.conv_layer.conv_layer(filter_count=2))
    cnn.train(trainX, trainY)
    # cnn.predict(x, y)
