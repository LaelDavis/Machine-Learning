
import numpy as np #linear algebra
"""
Our goal in training is to find the best set of weights
and biases that minimizes the loss function. 

We will compare our outcome array [expected value] (line 57) 
with out 'y' array [actual value] that we have stored through numpy. (line 51)

"""

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        #initializing the variables for use
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(y.shape)
    def feedforward(self):
        #feedforward used for getting the outputs
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
    def backprop(self):
        """
        application of the chain rule to find derivative
        of the loss function with respect to weight2 and 
        weights1
        """
        #backprop used to update the weights and biases
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
        """
        update the weights with derivative (slope) 
        of the loss function

        """
        self.weights1 += d_weights1
        self.weights2 += d_weights2

if __name__ == "__main__":
    #create to arrays as inputs (4 by 2 dimensions)
    X = np.array([[0,0,1],[0,1,1], [1,0,1], [1,1,1]])
    y = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork(X, y)
    #iterate 1500 times to try to match necessary accuracy and not cause overfitting
    for i in range(1500):
        nn.feedforward()
        nn.backprop()
    print(nn.output)