import numpy as np

class Linear:

    def __init__(self, num_inputs, num_outputs):
        """
        Initialize the weights to be zero-mean Gaussian with 
        variance 0.01 and biases to zero.

        :param num_inputs: Number of inputs to layer.
        :param num_outputs: Number of outputs after layer.
        """
        self.W = np.random.randn(num_inputs, num_outputs)*0.1
        self.b = np.zeros(num_outputs)

    def forward(self, A):
        """
        Forward operation of linear layer. Performs
        operation O = AW + b. Stores input to object.

        :param A: Input data matrix with rows as examples.
        :return O: Output data matrix after affine transformation.
        """
        self.A = A
        self.N = A.shape[0]
        O = A @ self.W + np.outer(np.ones(self.N), self.b)
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
        dOdb = np.ones(self.N)
        dOdA = self.W
        dLdW = dLdO.T @ dOdW
        dLdb = dLdO.T @ dOdb[:,None]
        dLdA = dLdO @ dOdA.T
        self.dLdW = dLdW.T
        self.dLdb = dLdb.flatten()
        return dLdA
