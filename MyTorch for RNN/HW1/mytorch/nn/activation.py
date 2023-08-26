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
        def sig(x):
            return 1/(1 + np.exp(-x))
        # sigfunc = np.vectorize(sig)
        newH = sig(self.H)
        return newH

    def backward(self):
        """
        Compute derivative of identity activation function.
        
        :return dAdH: Element-wise derivative with respect to
        input to activation function H.
        """
        def sig(x):
            f = 1/(1 + np.exp(-x)) 
            return f * (1 - f)
        # sigfunc = np.vectorize(sig)
        newH = sig(self.H)
        return newH


class Tanh:
    
    def forward(self, H):
        """
        Compute tanh activation function.
        
        :param H: Output from hidden or final layer.
        :return A: Output after applying activation function.
        """
        self.H = H
        def tanh(x):
            t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
            return t
        # tanhfunc = np.vectorize(tanh)
        newH = tanh(self.H)
        return newH

    def backward(self):
        """
        Compute derivative of identity activation function.
        
        :return dAdH: Element-wise derivative with respect to
        input to activation function H.
        """
        def tanh(x):
            t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
            return 1 - t**2

        # tanhfunc = np.vectorize(tanh)
        newH = tanh(self.H)
        return newH

class ReLU:

    def forward(self, H):
        """
        Compute tanh activation function.
        
        :param H: Output from hidden or final layer.
        :return A: Output after applying activation function.
        """
        self.H = H
        def relu(x):
            return max(0.0, x)
        relufunc = np.vectorize(relu)
        newH = relufunc(self.H)
        return newH

    def backward(self):
        """
        Compute derivative of identity activation function.
        
        :return dAdH: Element-wise derivative with respect to
        input to activation function H.
        """
        def relu(x):
            if x > 0:
                return 1
            else:
                return 0
        relufunc = np.vectorize(relu)
        newH = relufunc(self.H)
        return newH
