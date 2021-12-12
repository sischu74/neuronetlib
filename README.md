## An object-oriented approach to a neural network constructor
### 1. Introduction and motivation
In order to fully understand how artificial neural networks are trained, I wanted to write a constructor for feedforward neural networks, which creates a neural net only by having specified a list of the desired hidden layers.

I wrote the constructor from scratch, using only built-in python functions and numpy. The constructor follows an object-oriented approach and creates the specified layers and nodes as objects. These objects are stored in lists and iterated through or called by the various methods in the constructor class. The object oriented approach allows us to store values, derivative values, parent- and child nodes and so on for every node, which is very convenient for backpropagation and prediction, where we can pass values on node for node to go through the whole net.

The constructor supports different activation functions and tasks (regression and classification). For classification, the network uses the softmax function before the output layer and optimizes using the cross-entropy loss function. For regression, the loss function is the mean squared error.

### 2. How to access the constructor
#### 2.1 Create an instance of the class and pass the necessary arguments
To access the constructor, one has to define a variable, which is then an instance of the class 'ffnn':
```Python
model = ffnn([3,3]) # call the constructor and set the parameters
```

The arguments in brackets are:

1. List of the hidden layers (for example [20,20] for two hidden layers with 20 neurons each)
2. Task (regression or classification, default is regression)
3. Activation function (ReLU or softplus, default is ReLU)
4. Learning rate (step size for the adjustments in parameter learning. default is 0.1)
5. Batch size (how much of the dataset is used for mini-batch gradient descent
Below are all the possible arguments:

```Python
ffnn(h_layers, task = 'regression', activ_func = 'relu', learning_rate = 0.1, batch_size = 0.1)
```

#### 2.2 Construct the network
The model is now stored in a variable. In our case, the variable is called 'model'. We can now call the construct method on our model, which builds the specified layers and nodes. We need to specify both the x and y part of the dataset, so that the model knows how large the input and output layer need to be.

```Pyrhon
model.construct(x,y) # construct the network with the dataset (needs it for shape of input and output layer)
```

#### 2.3 Train the model
Now we can call the train method on our model to learn the parameters. The necessary arguments for this method are the whole dataset (x and y):

```Python
model.train(x,y) # train the network with training data
```

#### 2.4 predict on the test dataset
We now possess a trained model and can use it to make predictions on the test set with the 'predictions' method.

```Python
model.predictions(x,y) # predict on the test data with the trained model
```
