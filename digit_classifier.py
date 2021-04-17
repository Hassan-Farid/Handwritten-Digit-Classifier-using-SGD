from keras.datasets import mnist
import random
import numpy as np

#Loading images as training and testing data from MNIST dataset
training_data, testing_data = mnist.load_data()

#Defining a vector generator for our neural network
def vectorized_result(j):
  e = np.zeros((10,1)) #Creates a vector with (10 x 1) order
  e[j] = 1.0 #Set the particular index to 1 from the vector for the particular input
  return e

#Performing transformations to get the training and testing data for neural network
training_inputs = [np.reshape(x, (784 , 1)) for x in training_data[0]]
training_results = [vectorized_result(y) for y in training_data[1]]
train_data = list(zip(training_inputs , training_results))
test_inputs = [np.reshape(x, (784 , 1)) for x in testing_data[0]]
test_data = list(zip(test_inputs , testing_data[1]))

#Creating a class for our Neural Network to train and classify digital number images
class Network(object):
  
  def __init__(self,sizes):
    '''
      The constructor takes an argument "sizes" referring to the number of layers to be used for the network along with its number of neurons
    '''

    self.num_layers = len(sizes) #Length of inputted list will tell the total number of layers in the network
    self.sizes = sizes
    self.biases = [np.random.randn(y,1) for y in sizes[1:]] #Biases created randomly for all neurons presents in the active layers (hidden and output)
    self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])] #Assigning random weights to the neurons

  def feed_forward(self, sigma):
    '''
      Returns the output of the network if 'sigma' is served as the input. 
    '''

    for b,w in zip(self.biases, self.weights):
      sigma = sigmoid(np.dot(w, sigma) + b) #Computing the sigmoid activation function for the provided input
      return sigma

  def SGD(self, train_data, epochs, mini_batch_size, eta, test_data=None):
    '''
      Performs the stochastic gradient descent on a particular mini-batch of training data for provided number of epochs
    '''

    if test_data: #In case of validating results
      num_test = len(test_data)
    n = len(train_data) #Counting number of training data samples
    for j in range(epochs):
      random.shuffle(train_data) #Randomly shuffle the training data for each epoch

      mini_batches = [train_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)] #Creating mini-batches from the training data 
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, eta) #Update the mini_batch's weights and biases according to condition
      if test_data:
        print("Epoch {0}: {1}/{2}".format(j, self.evaluate(test_data), num_test)) #Training test data against itself after each epoch to get better results
      else:
        print("Epoch {0}: Complete".format(j))

  def update_mini_batch(self, mini_batch, eta):
    '''
      Takes a mini_batch and changes its weights and biases based on the result provided by applying SGD using backpropogation, using provided learning rate 
    '''

    new_b = [np.zeros(b.shape) for b in self.biases] #Computing dimensions of various baises
    new_w = [np.zeros(w.shape) for w in self.weights] #Computing dimensions of various weights

    for x,y in mini_batch:
      delta_new_b, delta_new_w = self.backprop(x,y) #Applying backpropogation to find delta values for biases and weights
      new_b = [nb + dnb for nb, dnb in zip(new_b, delta_new_b)] #Computing the new set of biases after applying backpropogation
      new_w = [nw + dnw for nw, dnw in zip(new_w, delta_new_w)] #Computing the new set of weights after applying backpropogation

    self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip (self.weights, new_w)] #Computing new weights based on current learning rate
    self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip (self.biases, new_b)] #Computing new biases based on current learning rate

  def backprop(self, x, y):
    '''
      Backpropogation algorithm that returns the gradient for cost function using tuple values
    '''

    new_b = [np.zeros(b.shape) for b in self.biases] #Computing dimensions of various baises
    new_w = [np.zeros(w.shape) for w in self.weights] #Computing dimensions of various weights

    activation = x #Initializing the activation value with provided input value x
    activations = [x] #List to store all the activations, layer by layer

    z_list = [] #List to store the z-vectors, for all layers, layer by layer

    for b,w in zip(self.biases, self.weights):
      z = np.dot(w, activation) + b #Computing the value of z for the sigmoid function
      z_list.append(z) #Appending obtained value to the list of z-vectors
      activation - sigmoid(z) #Computing the sigmoid function for the obtained z-vector
      activations.append(activation) #Appending obtained value of activation to the list of activations

    delta = self.cost_derv(activations[-1], y) * sigmoid_prime(z_list[-1]) #Computing delta value for backward pass
    new_b[-1] = delta #Using delta value to compute the final bias
    new_w[-1] = np.dot(delta, activations[-2].transpose()) #Using delta value to compute the final weight

    for l in range(2, self.num_layers): #Since the number of layers can extend from 2 (input, output) to provided number
      z = z_list[-1] #Assigning the last stored z-vector in the z_list
      sp = sigmoid_prime(z) #Computing the derivative of the sigmoid method on the obtained z-vector
      delta = np.dot(self.weights[-l+1].transpose(), delta) * sp #Computing new value of delta for each layer
      new_b[-1] = delta
      new_w[-1] = np.dot(delta, activations[-2].transpose())

    return (new_w, new_b) #Returns the tuple of weights and biases representing gradient for the cost function C(x)

  def evaluate(self, test_data):
    '''
      Returns the set of test inputs for which neural network provides the correct result
    '''

    test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
    return sum(int(x == y) for (x,y) in test_results)

  def cost_derv(self, output_activations, y):
    '''
      Returns vector of partial derivatives for the cost function
    '''
    return (output_activations - y)

def sigmoid(z):
  '''
    Returns the value for sigmoid function applied to a z-vector
  '''

  return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
  '''
    Returns the value of derivative of sigmoid function applied to a z-vector
  '''

  return sigmoid(z) * (1 - sigmoid(z))

#Creating a neural net and training our data
net = Network([784, 30, 10])
net.SGD(train_data, 30, 10, 3.0, test_data = test_data)
