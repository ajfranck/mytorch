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
        self.N = O.shape[0]
        # Compute mean squared error, then normalize by number of elements in O.
        #L = None was original
        L = np.sum((O - Y)**2) / (self.N * O.shape[1])
        return L
    
    def backward(self):
        """
        Compute gradient dLdO for MSE loss.

        :return dLdO: Gradient of loss with respect to output O.
        """
        O = self.O
        Y = self.Y
        dLdO = 2 * (O - Y) / (self.N * O.shape[1])
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
        #first do softmax of predicted and targets
        O = np.exp(O) / np.sum(np.exp(O), axis=1, keepdims=True)
        Y = np.exp(Y) / np.sum(np.exp(Y), axis=1, keepdims=True)
        #compute MSE with rspect to true targets. Normalize by dividing by N only.
        L = (-np.sum(Y * np.log(O))) / self.N
        return L

    def backward(self):
        """
        Compute gradient dLdO for cross entropy loss.

        :return dLdO: Gradient of loss with respect to output O.
        """
        dLdO = -self.Y / self.O
        return dLdO
