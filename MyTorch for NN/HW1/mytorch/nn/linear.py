import numpy as np

class Linear:

    def __init__(self, num_inputs, num_outputs):
        """
        Initialize the weights to be zero-mean Gaussian with 
        variance 0.01 and biases to zero.

        :param num_inputs: Number of inputs to layer.
        :param num_outputs: Number of outputs after layer.
        """

        #intialize W using a normal distribution with variance 0.01 and mean 0
        self.W = np.random.normal(0, 0.01, (num_inputs, num_outputs))

        #initialize b to be a vector of zeros
        self.b = np.zeros((num_outputs, 1))
        


    def forward(self, A):
        """
        Forward operation of linear layer. Performs
        operation O = AW + b. Stores input to object.

        :param A: Input data matrix with rows as examples.
        :return O: Output data matrix after affine transformation.
        """
        self.A = A
        self.N = A.shape[0]
        ones = np.ones((1,self.N))
        self.b = np.reshape(np.atleast_2d(self.b), (1, -1))
        # pt1 = ones.T @ np.atleast_2d(self.b)
        # pt2 = self.A @ self.W
        O = (ones.T @ np.atleast_2d(self.b)) + (self.A @ self.W)
        return O
        
    def backward(self, dLdO):
        """
        Backpropagation operation for variables in linear
        layer. Stores derivatives dLdW, dLdb and returns dLdA.

        :param dLdO: Derivative of loss with respect to output.
        Obtained from backward operation on loss object.
        :returns dLdA: Derivative of loss with respect to input.
        """
        dOdW = self.A
        dOdb = np.ones((self.N, 1))
        dOdA = self.W
        dLdW = dLdO.T @ dOdW
        dLdb = dLdO.T @ dOdb
        dLdA = dLdO @ dOdA.T
        self.dLdW = dLdW.T
        self.dLdb = dLdb.flatten()
        return dLdA
