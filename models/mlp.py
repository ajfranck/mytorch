import numpy as np

from mytorch.nn.linear import Linear
from mytorch.nn.activation import Identity, ReLU


class MLP0:

    def __init__(self, num_inputs, num_outputs):
        """
        Initialize MLP object with a single linear layer
        followed by an identity activation function.

        :param num_inputs: Number of inputs to layer.
        :param num_outputs: Number of outputs after layer.
        """
        self.layers = [Linear(num_inputs, num_outputs)]
        self.f = [Identity()]

    def forward(self, X):
        """
        Forward operation of MLP with zero hidden layers.

        :param X: Input data matrix with rows as examples.
        :return A1: Output data matrix after affine transformation
        and activation function.
        """
        H1 = self.layers[0].forward(X)
        A1 = self.f[0].forward(H1)
        return A1 

    def backward(self, dLdA1):
        """
        Backpropagation operation for MLP with zero hidden layers.
        Performs backpropagation on appropriate layers to obtain
        gradient with respect to the input X.
        Does not return anything.

        :param dLdA1: Derivative of loss with respect to output A1.
        Obtained from backward operation on loss object.
        """
        dA1dH1 = self.f[0].backward()
        dLdH1 = dLdA1 * dA1dH1
        dLdX = self.layers[0].backward(dLdH1)
       

class MLP1:

    def __init__(self, num_inputs, num_outputs, num_hiddens):
        """
        Initialize MLP object with a single hidden layer
        followed by a ReLU activation function. Use and Identity
        activation function at the output.

        :param num_inputs: Number of inputs to model.
        :param num_outputs: Number of outputs from model.
        :param num_hiddens: Size of hidden layer.
        """
        self.layers = [Linear(num_inputs, num_hiddens), Linear(num_hiddens, num_outputs)]
        self.f = [ReLU(), Identity()]

    def forward(self, X):
        """
        Forward operation of MLP with one hidden layer.

        :param X: Input data matrix with rows as examples.
        :return A2: Output data matrix.
        """
        H1 = self.layers[0].forward(X)
        A1 = self.f[0].forward(H1)
        H2 = self.layers[1].forward(A1)
        A2 = self.f[1].forward(H2)
        return A2

    def backward(self, dLdA2):
        """
        Backpropagation operation for MLP with one hidden layer.
        Performs backpropagation on appropriate layers to obtain
        gradient with respect to the input X.
        Does not return anything.

        :param dLdA2: Derivative of loss with respect to output A2.
        Obtained from backward operation on loss object.
        """
        dA2dH2 = self.f[1].backward()
        dLdH2 = dLdA2 * dA2dH2
        dLdA1 = self.layers[1].backward(dLdH2)

        dA1dH1 = self.f[0].backward()
        dLdH1 = dLdA1 * dA1dH1
        dLdX = self.layers[0].backward(dLdH1)


class MLP4:
    
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        """
        Initialize 4 hidden layers and an output layer of shape below:
        Layer1 (num_inputs, num_hiddens),
        Layer2 (num_hiddens, num_hiddens),
        Layer3 (num_hiddens, num_hiddens),
        Layer4 (num_hiddens, num_hiddens),
        Output Layer (num_hiddens, num_outputs)
        Follow all hidden layers with a ReLU activation function.

        :param num_inputs: Number of inputs to model.
        :param num_outputs: Number of outputs from model.
        :param num_hiddens: Size of hidden layer.
        """
        self.layers = [Linear(num_inputs, num_hiddens), Linear(num_hiddens, num_hiddens),
                Linear(num_hiddens, num_hiddens), Linear(num_hiddens, num_hiddens),
                Linear(num_hiddens, num_outputs)]
        self.f = [ReLU(), ReLU(), ReLU(), ReLU(), Identity()]

    def forward(self, X):
        """
        Forward operation of MLP with four hidden layers.

        :param X: Input data matrix with rows as examples.
        :return A: Output data matrix.
        """
        L = len(self.layers)
        A = X
        for i in range(L):
            H = self.layers[i].forward(A)
            A = self.f[i].forward(H)
        return A

    def backward(self, dLdA):
        """
        Backpropagation operation for MLP with four hidden layers.
        Performs backpropagation on appropriate layers to obtain
        gradient with respect to the input X.
        Does not return anything.

        :param dLdA: Derivative of loss with respect to output A.
        Obtained from backward operation on loss object.
        """
        L = len(self.layers)
        for i in reversed(range(L)):
            dAdH = self.f[i].backward()
            dLdH = dLdA * dAdH
            dLdA = self.layers[i].backward(dLdH)
