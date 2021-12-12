class layer():
    def __init__(self, layer_nr, nodes):
        self.layer_nr = layer_nr # position of the layer
        self.nodes = nodes # list of the nodes belonging to this layer

    # method returns a list of the node values in the layer
    def value_list(self):
        values = []
        for neuron in self.nodes:
            values.append(neuron.value)
        return values

class node():
    def __init__(self, w, b=0, child_nodes=[], parent_nodes=[], value=0, derivative=0):
        self.w = w # list of this node's weight parameters
        self.b = b # this node's bias parameter
        self.child_nodes = child_nodes # list of this node's child nodes
        self.parent_nodes = parent_nodes # list of this node's parent nodes
        self.value = value # calculated value of this node. used for prediction and backpropagation
        self.output = 0 # variable storing the softmax likelihood output (for classification)
        self.derivative = derivative # derivative of loss function with respect to this node. calculated in 'train' procedure

class ffnn():
    def __init__(self, h_layers, task = 'regression', activ_func = 'relu', learning_rate = 0.1, batch_size = 0.1):
        self.h_layers = h_layers
        self.task = task
        self.activ_func = activ_func
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.layers = 0 # layers are later added by the 'construct' method

    def predict(self, x, y):
        layers = self.layers
        error = 0
        preds = []
        global denominator

        for sample in range(x.shape[0]):
            # fill in the values of the input layer with the feature values
            for i in range(x.shape[1]):
                layers[0].nodes[i].value = x[sample][i]

            # calculate all node values as the dot product of its w vector and the parent nodes' values plus a bias 'b'
            for i in range(1,len(layers)): # cycle through layers
                for neuron in layers[i].nodes: # cycle through nodes in layer
                    neuron.value = np.dot(neuron.w, layers[i-1].value_list()) + neuron.b # neuron value is the dot product between parent node values and weight parameters plus bias
                    if (self.task == 'classification') & (layers[-1].layer_nr == i): # don't apply activation function to the last layer for classification
                        pass  
                    elif self.activ_func == 'relu':
                        neuron.value = max(0, neuron.value)
                    elif self.activ_func == 'softplus':
                        neuron.value = np.log(1+np.exp(neuron.value))

            # apply softmax function to last layer for classification
            if self.task == 'classification':
                denominator = 0
                for others in range(len(layers[-1].nodes)): # cycle through other nodes in the last layer
                    denominator += np.exp(layers[-1].nodes[others].value)
                for neuron in range(len(layers[-1].nodes)): # cycle through output nodes
                    layers[-1].nodes[neuron].output = np.exp(layers[-1].nodes[neuron].value)/denominator
                    print(layers[-1].nodes[neuron].value)
                    print(layers[-1].nodes[neuron].output)
                    print(denominator)

            # error calculation for regression
            if self.task == 'regression':
                preds.append(layers[-1].nodes[0].value)
                error += (y[sample] - float(layers[-1].nodes[0].value))**2

            # error calculation for classification
            elif self.task == 'classification':
                # predict the maximum value in the output layer
                output_values = layers[-1].value_list()
                preds.append(output_values.index(max(output_values)))
                if preds[-1] != y[sample]:
                    error += 1
                    
        return error, preds

    def predictions(self, x, y):
        # function is only for user access and prints the figures nicely.
        # basically a pretty-print of the 'internal' predict function above
        errors, preds = self.predict(x,y)
        print('Test Error:', errors/x.shape[0])
        print('Predicted values:', preds)

    def construct(self, x, y):
        # construct the layers and nodes specified
        global node # tell python that we mean the global class "node" and not a local variable
        layers = [] # create an empty list that we fill with layers containing the nodes

        # create an input layer with d nodes. these don't contain any weights
        layers.append(layer(0,[]))
        for i in range(x.shape[1]): # one input node for every feature the input data contains
            layers[0].nodes.append(node([]))

        # create the hidden layers
        for i in range(1,len(self.h_layers)+1):
            layers.append(layer(i,[])) # create a layer object
            for j in range(self.h_layers[i-1]): # create one hidden node per specified
                layers[i].nodes.append(node([]))

        # create an output layer
        if self.task == 'classification': # output node contains as many parameters as the last hidden layer contains nodes
            layers.append(layer(len(layers),[]))
            for i in range(len(np.unique(y))): # one node for every unique y value (each category)
                layers[-1].nodes.append(node(np.ones(self.h_layers[-1])))
        elif self.task == 'regression':
            layers.append(layer(len(layers), list([node(np.ones(self.h_layers[-1]))]))) # for regression, we only need one output node

        # fill in the child nodes for every node
        for i in range(len(layers)-1): # cycle through all layers but the output layer
            for neuron in layers[i].nodes: # cycle through every node in that layer
                neuron.child_nodes = neuron.child_nodes + layers[i+1].nodes # because network is fully connected, append every node in the next layer as a child node

        # fill in the parent nodes for every node
        for i in range(1,len(layers)):
            for neuron in layers[i].nodes:
                neuron.parent_nodes = neuron.parent_nodes + layers[i-1].nodes

        # initialize the w parameters to random float between [-5,5]
        for i in range(1, len(self.h_layers)+1):
            for neuron in layers[i].nodes:
                neuron.w.extend(np.random.uniform(-0.5,0.5,len(layers[i-1].nodes)))

        self.layers = layers
        return layers

    def train(self, x, y):
        # use the global classes layer and node
        global layer
        global node
        global denominator

        iteration = 0 # counter for the iterations
        learning_rate = self.learning_rate # get learning rate from the class attributes

        # concatenate x and y together to randomly sample from it later
        mini_batch = np.concatenate((x,y),1)

        for i in range(1):
       # while self.early_stopping(x,y,iteration) == False: # check for convergence and early stopping criteria
            # randomly select subset of the data for mini-batch stochastic gradient descent
            np.random.shuffle(mini_batch) # shuffle the data
            mini_batch = mini_batch[:int(x.shape[0]*self.batch_size),:]

            layers = self.layers # get the updated layers from the class attributes
            for sample in range(mini_batch.shape[0]):
                # predict on the current point. the predict function also calculates all node values for this training point
                error, prediction = self.predict(np.array(mini_batch[sample,:-1]).reshape(1,mini_batch[sample,:-1].shape[0]),np.array(mini_batch[sample,-1]).reshape(1,1))
                # calculate partial derivatives for nodes in the last layer
                if self.task == 'regression':
                    layers[-1].nodes[0].derivative = (-2/mini_batch.shape[0])*(mini_batch[sample,1]-neuron.value)
                elif self.task == 'classification':
                    for neuron in range(len(layers[-1].nodes)):
                        if neuron == mini_batch[sample,1]: # derivative of the correct label node is different
                            layers[-1].nodes[neuron].derivative = -(np.exp(layers[-1].nodes[neuron].value)*denominator-(np.exp(2*layers[-1].nodes[neuron].value)))/(layers[-1].nodes[neuron].output*(denominator)**2)
                        else:
                            layers[-1].nodes[neuron].derivative = np.exp(layers[-1].nodes[neuron].value)*np.exp(layers[-1].nodes[int(mini_batch[sample,1])].value)/(layers[-1].nodes[int(mini_batch[sample,1])].output * (denominator)**2)

                # calculate partial derivatives of every node based on its child node
                for layer in range(2,len(layers)): # cycle backwards through 2nd last to second layer (we don't need derivative of the input layer)
                    for neuron in range(len(layers[-layer].nodes)): # cycle through nodes per layer
                        for child in range(len(layers[-layer].nodes[neuron].child_nodes)): # cycle through child nodes per node
                            if (self.task == 'classification') & (layer == 2): # for 2nd last layer we have different derivatives, due to the softmax on output layer
                                layers[-layer].nodes[neuron].derivative = layers[-layer].nodes[neuron].child_nodes[child].w[neuron] * layers[-layer].nodes[neuron].child_nodes[child].derivative
                            else:
                                if self.activ_func == 'relu':
                                    if layers[-layer].nodes[neuron].child_nodes[child].value >= 0: # if statement because of ReLU activation function
                                        layers[-layer].nodes[neuron].derivative = layers[-layer].nodes[neuron].child_nodes[child].w[neuron] * layers[-layer].nodes[neuron].child_nodes[child].derivative
                                    else:
                                        layers[-layer].nodes[neuron].derivative = 0
                                elif self.activ_func == 'softplus':
                                    layers[-layer].nodes[neuron].derivative = (layers[-layer].nodes[neuron].child_nodes[child].derivative * layers[-layer].nodes[neuron].child_nodes[child].w[neuron] * np.exp(layers[-layer].nodes[neuron].child_nodes[child].value))/(1+np.exp(layers[-layer].nodes[neuron].child_nodes[child].value))    

                # get the partial derivatives of parameters based on partial derivatives of nodes
                for layer in range(1,len(layers)): # cycle from last to second layer (input layer doesn't have parameters)
                    for neuron in range(len(layers[-layer].nodes)): # cycle through nodes per layer
                        if (self.task == 'classification') & (layer == 2): # for 2nd last layer in classification we have different derivatives, due to the softmax on output layer
                            layers[-layer].nodes[neuron].b -= learning_rate * layers[-layer].nodes[neuron].derivative
                            for weight in range(len(layers[-layer].nodes[neuron].w)): # cycle through weight per node
                                layers[-layer].nodes[neuron].w[weight] -= learning_rate * layers[-layer].nodes[neuron].parent_nodes[weight].value
                        else:
                            if self.activ_func == 'relu':
                                if layers[-layer].nodes[neuron].value >= 0: # if statement because of ReLU activation function
                                    layers[-layer].nodes[neuron].b -= learning_rate * layers[-layer].nodes[neuron].derivative
                            elif self.activ_func == 'softplus':
                                layers[-layer].nodes[neuron].b -= (learning_rate * layers[-layer].nodes[neuron].derivative * np.exp(layers[-layer].nodes[neuron].value))/(1+np.exp(layers[-layer].nodes[neuron].value))
                            for weight in range(len(layers[-layer].nodes[neuron].w)): # cycle through weight per node
                                if self.activ_func == 'relu':
                                    if layers[-layer].nodes[neuron].value >= 0: # if statement because of ReLU activation function
                                        layers[-layer].nodes[neuron].w[weight] -= learning_rate * layers[-layer].nodes[neuron].parent_nodes[weight].value * layers[-layer].nodes[neuron].derivative
                                elif self.activ_func == 'softplus':
                                    layers[-layer].nodes[neuron].w[weight] -= (learning_rate * layers[-layer].nodes[neuron].parent_nodes[weight].value * np.exp(layers[-layer].nodes[neuron].value) * layers[-layer].nodes[neuron].derivative)/(1+np.exp(layers[-layer].nodes[neuron].value))
            
            print('iteration done')
            iteration += 1 # increment the iteration counter
        
        errors, preds = self.predict(x,y) # predict with the final parameters
        print('Iterations until stopped:', iteration)
        print('Training error (per sample):', errors/x.shape[0])
        return self.layers

    def early_stopping(self, x, y, iteration):
        # use the global variables
        global last_error
        global test_error
        global old_layers
        global return_counter
        error, predictions = self.predict(x,y) # predict on the test data

        # check if algorithm has converged
        if iteration == 0: # if it's the first iteration, skip everything here right away
            return_counter = 0 # variable counts how many times we returned back to the old model
            test_error = [999999999999999] # insert large number, so that the error will be smaller for the first iteration
            last_error = test_error[0] # error of last iteration to compare if it changed (convergence check)
            return False
        else:
            if np.sqrt((error-last_error)**2) < 0.000001: # if error barely changes with an iteration, declare convergence
                if iteration > 1000:
                    print('...algorithm converged!')
                    return True
                else:
                    print('...converged, but not enough iterations')

        last_error = error

        # check if test error has decreased over the last 50 iterations
        if iteration % 50 == 0: # check only every 50'th iteration
            test_error.append(error)
            if test_error[-1] < test_error[-2]: # check if error is less than last time
                old_layers = self.layers # safe network in case the next one performs worse
                return False
            else:
                print('...returned to old parameters...')
                return_counter += 1 # increment return-to-old-model-count
                if return_counter > 20:
                    return True
                self.layers = old_layers # continue training with old network if new performs worse
                return False
        else:
            return False

model = ffnn([3,3], activ_func = 'softplus', learning_rate = 0.1, batch_size = 1) # call the constructor and set the parameters
model.construct(x) # construct the network with the input data (needs it for shape of input layer)
model.train(x,y) # train the network with training data
model.predictions(x,y) # predict on the test data with the trained model
