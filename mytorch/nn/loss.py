import numpy as np

class MSELoss:
    
    def forward(self, O, Y):
        """
        Compute MSE loss between outputs O and true targets Y.

        :param O: Output predictions.
        :param Y: True targets.
        :return L: Mean squared error, normalized by total number
        of elements in O.
        """
        self.O = O
        self.Y = Y
        self.dims = O.shape
        self.N = O.shape[0]
        self.q = O.shape[1]
        L = np.sum((O - Y)**2) / np.prod(self.dims)
        return L
    
    def backward(self):
        """
        Compute gradient dLdO for MSE loss.

        :return dLdO: Gradient of loss with respect to output O.
        """
        O = self.O
        Y = self.Y
        dLdO = 2*(O - Y) / np.prod(self.dims)
        return dLdO
    
class CrossEntropyLoss:

    def forward(self, O, Y):
        """
        Compute cross entropy loss between outputs O and true targets Y
        as well as softmax probabilities for outputs O.
        Note: Does not match PyTorch unless Y is a one-hot label matrix.

        :param O: Output predictions.
        :param Y: True targets.
        :return L: Cross entropy loss, normalized by number of examples.
        """
        self.O = O
        self.Y = Y
        self.N = O.shape[0]
        O_exp = np.exp(O)
        partition = O_exp.sum(1, keepdims=True)
        self.softmax = O_exp / partition
        L = -np.sum(Y * np.log(self.softmax)) / self.N
        return L

    def backward(self):
        """
        Compute gradient dLdO for cross entropy loss.

        :return dLdO: Gradient of loss with respect to output O.
        """
        dLdO = (self.softmax - self.Y) / self.N
        return dLdO
