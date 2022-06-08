class layer(): # layer class
    def __init__(self, w, b, size):
        self.w = w # w matrix (j'th row stores a row vector of weights of j'th neuron)
        self.b = b # b vector (b[j] is the bias term of the j'th neuron in this layer)
        self.size = size # number of neurons in this layer
        self.values = np.empty # vector of values
        self.output = np.empty # vector of output values
        self.derivatives = np.empty # vector of derivatives
        
class ffnn(): # neural network class
    def __init__(self, h_layers, learning_rate = 0.1, batch_size = 0.1, targetAccuracy = 0.9):
        self.h_layers = h_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size # share of the dataset used to update params
        self.layers = np.empty # layers are later added by the 'construct' method
        self.targetAccuracy = targetAccuracy # when this accuracy (plus sufficient convergence) is achieved, training is halted
        
    def __construct(self, d, k = 1): # private constructor method to create layers and initialize params
        # d is number of features
        # k is number of classes in classification
        
        self.layers = np.empty(len(self.h_layers) + 2, layer) # initialize array of layers to empty array of dimension h + 2 (h is number of hidden layers)
        self.layers[0] = layer(np.empty, np.zeros(d), d) # initialize input layer with random b vector

        for h in range(len(self.h_layers)): # create hidden layers
            self.layers[h+1] = layer(np.random.rand(self.h_layers[h], self.layers[h].size) - 0.5, np.random.rand(self.h_layers[h]) - 0.5, self.h_layers[h])
        
        # output layer
        self.layers[-1] = layer(np.random.rand(k, self.layers[-2].size) - 0.5, np.random.rand(k) - 0.5, k)
    
    def __randomParams(self): # private method for setting new random weight matrix for all layers
        for i in range(1, len(self.h_layers)+2):
            self.layers[i].w = np.random.rand(self.layers[i].w.shape[0], self.layers[i].w.shape[1])-0.5
    
    def train(self, x, y):
        iters = 0
        eta = self.learning_rate
        self.__construct(x.shape[1], len(np.unique(y))) # call internal constructor method
        
        data = np.concatenate((x, y), 1) # merge x and y data such that the data/labels correspond
        batchSize = int(self.batch_size * x.shape[0])
        
        while(not self.__earlyStopping(x,y,iters)):
            # randomly select batch to train on
            index = np.random.randint(x.shape[0], size = batchSize) # create random index set to sample
            batch = data[index,:] # create random sample from the data
            
            for sample in range(batchSize): # iterate through all samples in the batch
                point = batch[sample, :-1]
                label = batch[sample, -1]
                self.__forwardPass(point, label) # call internal prediction method
                
                # compute derivative with respect to nodes in last layer
                self.layers[-1].derivatives = (self.layers[-1].values - self.__oneHot(label))/batchSize
                # change w and b
                self.layers[-1].w -= np.dot(np.reshape(self.layers[-1].derivatives, (self.layers[-1].size, 1)), np.reshape(self.layers[-2].values.T, (1,self.layers[-2].size)))*eta
                self.layers[-1].b -= self.layers[-1].derivatives * eta
    
                for i in range(len(self.layers)-2, 0, -1):
                    self.layers[i].derivatives = np.dot(self.layers[i+1].w.T, self.layers[i+1].derivatives)
                    self.layers[i].b -= self.layers[i].derivatives * eta
                    self.layers[i].w -= np.dot(np.reshape(self.layers[i].derivatives, (self.layers[i].size, 1)), np.reshape(self.layers[i-1].values.T, (1,self.layers[i-1].size)))*eta
                        
            iters += 1
            print("iterations: ", iters)
                    
    def __oneHot(self, x):
        vect = np.zeros(self.layers[-1].size)
        vect[int(x)] = 1
        return vect
    
    def __forwardPass(self, x, y): # accepts x vector and y label (forward pass for one label)
        self.layers[0].values = x
        for i in range(1, self.layers.size): # compute layer values as matmuls
            layer = self.layers[i]
            layer.values = np.dot(layer.w, self.layers[i-1].values) + layer.b
            if (i != (self.layers.size - 1)): layer.values[layer.values < 0] = 0 # apply ReLU
        self.layers[-1].values = self.__softmax(self.layers[-1].values)
        return np.argmax(self.layers[-1].values) # predict the maximum value in the output layer
    
    def __softmax(self, x):
        maxi = np.max(x) # normalize the values to prevent overflows
        sumAll = sum(np.exp(x-maxi))
        return np.exp(x-maxi)/sumAll
    
    def predict(self, x, y): # public prediction method
        preds = np.empty(x.shape[0],int)
        for i in range(x.shape[0]):
            preds[i] = self.__forwardPass(x[i], y[i])
        acc = self.__accuracy(preds, y)
        return preds, acc
            
    def __earlyStopping(self, x, y, iteration): # decides when to stop the training
        global oldAccuracy
        if (iteration % 10 != 0): return False 
        if iteration == 0: # if it's the first iteration, skip everything here right away
            last_error = 999999999999999 # error of last iteration to compare if it changed (convergence check)
            oldAccuracy = 0.3 # set accuracy to 30%
            return False
        else: # check only every 10'th iteration
            data = np.concatenate((x, y), 1) # merge x and y so that they correspond (to sample from it)
            index = np.random.randint(x.shape[0], size = int(x.shape[0]*0.1)) # create random index set to sample
            sample = data[index,:] # create random sample from the data
            predictions = self.predict(sample[:,:-1], sample[:,-1])[0] # predict on the sample
            accuracy = self.__accuracy(predictions, sample[:, -1])
            print("current accuracy: ", accuracy)
            if ((accuracy < oldAccuracy) & (accuracy < 0.3)): # if accuracy decreased and is less than 30%, put new random parameters
                print("new random params")
                self.__randomParams()
                oldAccuracy = 0.3
                return False
            # if accuracy barely changes and achieved target accuracy, stop training
            if ((abs(accuracy - oldAccuracy) < 0.001) & (accuracy > self.targetAccuracy)): return True
            oldAccuracy = accuracy
            return False
    
    def __accuracy(self, preds, y):
        if (y.size == 0): raise Exception("empty sample")
        errors = 0
        for i in range (y.size):
            if (preds[i] != int(y[i])): errors += 1
        return 1 - errors/y.size
