import numpy as np

class Identity:

    def forward(self, H):
        """
        Compute identity activation function.
        
        :param H: Output from hidden or final layer.
        :return A: Output after applying activation function.
        """
        self.H = H
        return self.H

    def backward(self):
        """
        Compute derivative of identity activation function.
        
        :return dAdH: Element-wise derivative with respect to
        input to activation function H.
        """
        dAdH = np.ones(self.H.shape, dtype="f")
        return dAdH


class Sigmoid:
 
    def forward(self, H):
        """
        Compute sigmoid activation function.
        
        :param H: Output from hidden or final layer.
        :return A: Output after applying activation function.
        """
        self.H = H
        return 1 / (1 + np.exp(-H))

    def backward(self):
        """
        Compute derivative of identity activation function.
        
        :return dAdH: Element-wise derivative with respect to
        input to activation function H.
        """
        eH = np.exp(-self.H)
        dAdH = eH / (1 + eH)**2
        return dAdH

class Tanh:
    
    def forward(self, H):
        """
        Compute tanh activation function.
        
        :param H: Output from hidden or final layer.
        :return A: Output after applying activation function.
        """
        self.H = H
        return np.tanh(H)

    def backward(self):
        """
        Compute derivative of identity activation function.
        
        :return dAdH: Element-wise derivative with respect to
        input to activation function H.
        """
        dAdH = 1 - np.tanh(self.H)**2
        return dAdH

class ReLU:

    def forward(self, H):
        """
        Compute tanh activation function.
        
        :param H: Output from hidden or final layer.
        :return A: Output after applying activation function.
        """
        self.H = H
        return np.maximum(0, H)

    def backward(self):
        """
        Compute derivative of identity activation function.
        
        :return dAdH: Element-wise derivative with respect to
        input to activation function H.
        """
        dAdH = (self.H > 0).astype("f")
        return dAdH
