import numpy as np

class SGD:

    def __init__(self, model, lr=0.1):
        """
        Initialize SGD object.

        :param model: Neural network object from mytorch.nn.
        :param lr: Learning rate.
        """
        self.model = model
        self.lr = lr
        # for use with MLP, which has multiple layers
        if hasattr(model, "layers"):
            self.l = model.layers
            self.L = len(model.layers)

    def step(self):
        """
        Perform a single SGD step.
        """
        if hasattr(self.model, "layers"):
            for i in range(self.L):
                dLdW = self.l[i].dLdW
                dLdb = self.l[i].dLdb
                self.l[i].W -= self.lr * dLdW
                self.l[i].b -= self.lr * dLdb
        else:
            dLdW = self.model.dLdW
            dLdb = self.model.dLdb
            self.model.W -= self.lr * dLdW
            self.model.b -= self.lr * dLdb    

    def zero_grad(self):
        """
        Dummy function for use with d2l library.
        """
        return
