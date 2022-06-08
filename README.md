## Feedforward neural network library from scratch
### 1. Introduction and motivation
In order to fully understand how artificial neural networks are trained, I wanted to write a constructor for feedforward neural networks, which creates a neural net only by having specified a list of the desired hidden layers and some custom parameters, like learning rate and batch size for gradient descent.

I wrote the constructor from scratch, using only built-in python functions and NumPy. The training was optimized to make great use of vectorization and especially the highly optimized NumPy matrix multiplication. The constructor class stores a list of layers, which are their own class and store a weight matrix and various vectors for derivatives, intercepts and values each.

For classification, the network uses the softmax function before the output layer and optimizes using the cross-entropy loss function.

The constructor is demoed on the MNIST dataset, where it achieved 89.2% accuracy after a few minutes of training. The Jupyter Notebook file can be found in the main branch.

### 2. How to access the constructor
#### 2.1 Create an instance of the class and pass the necessary arguments
To access the constructor, one has to define a variable, which is then an instance of the class 'ffnn':
```Python
model = ffnn([100,70]) # call the constructor and set the parameters. In this case two hidden layers with 100 and 70 nodes each.
```

The arguments in brackets are:

1. List of the hidden layers (for example [20,20] for two hidden layers with 20 neurons each)
2. Learning rate (step size for the adjustments in parameter learning. default is 0.04)
3. Batch size (how much of the dataset is used for mini-batch gradient descent)
4. Target Accuracy (constructor will stop training once this accuracy and sufficient convergence is achieved)
Below are all the possible arguments:

```Python
model = ffnn(h_layers = [100,70], learning_rate = 0.05, batch_size = 0.1, targetAccuracy = 0.92)
```

#### 2.2 Construct the network
The model is now stored in a variable. In our case, the variable is called 'model'. We can now call the construct method on our model, which builds the specified layers and nodes. We need to specify both the x and y part of the dataset, so that the model knows how large the input and output layer need to be.

```Pyrhon
model.construct(x,y) # construct the network with the dataset (needs it for shape of input and output layer)
```

#### 2.3 Train the model
Now we can call the train method on our model to learn the parameters. The necessary arguments for this method are the whole dataset (x and y). Make sure that the dimensions of the y vector is n * 1 (if the numpy array has shape (n,) instead of (n,1) you need to reshape it).

```Python
model.train(x,y) # train the network with training data
```

#### 2.4 predict on the test dataset
We now possess a trained model and can use it to make predictions on the test set with the 'predict' method. This will return an array with the predictions as well as the accuracy of the model on the training data.

```Python
model.predict(x,y) # predict on the test data with the trained model
```
