import numpy as np

class Conv2d:
    
    def __init__(self, in_channels, out_channels, kernel_size):
        """
        Initialize the weights to be zero-mean Gaussian. You do not need to include
        a bias term.

        :param num_inputs: Number of inputs to layer.
        :param num_outputs: Number of outputs after layer.
        """
        #initalize weights to be zero-mean Gaussian
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.W = np.random.normal(0, 1, (out_channels, in_channels, kernel_size, kernel_size))


    def forward(self, X):
        """
        Forward operation of convolutional layer.

        :param X: Input image batch of size (batch_size, in_channels, height, width).
        :return O: Output feature map.
        """
        #get dimensions of input
        batch_size, in_channels, height, width = X.shape
        #get dimensions of kernel
        out_channels, in_channels, kernel_size, kernel_size = self.W.shape
        #calculate output dimensions
        out_height = height - kernel_size + 1
        out_width = width - kernel_size + 1
        # reshape input and kernel for efficient computation
        X_reshaped = X.reshape(batch_size, in_channels, out_height, out_width, kernel_size, kernel_size)
        W_reshaped = self.W.reshape(1, out_channels, in_channels, kernel_size, kernel_size)
    
        # perform convolution using dot product and sum along appropriate axes
        O = np.sum(X_reshaped * W_reshaped, axis=(2, 4, 5))
    
        return O 
    
    def backward(self, dLdO):
        """
        Backward operation of convolutional layer. Stores derivative dLdW, and returns
        dLdX.

        :param dLdO: Derivative of loss with respect to output.
        Obtained from backward operation on loss object.
        :returns dLdX: Derivative of loss with respect to input.
        """

        # Get dimensions of input
        batch_size, in_channels, height, width = self.X.shape
        # Get dimensions of kernel
        out_channels, in_channels, kernel_size, kernel_size = self.W.shape
        # Calculate output dimensions
        out_height = height - kernel_size + 1
        out_width = width - kernel_size + 1
        # Initialize output
        dLdX = np.zeros((batch_size, in_channels, height, width))
        # Initialize dLdW
        dLdW = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        # Reshape arrays for broadcasting
        X_reshaped = self.X.reshape(batch_size, in_channels, out_height, 1, out_width, 1)
        W_reshaped = self.W.reshape(1, out_channels, in_channels, kernel_size, kernel_size)
        # Calculate dLdW
        dLdW = np.sum(dLdO[:, :, :, :, np.newaxis, np.newaxis] * X_reshaped[:, np.newaxis, :, :, :, :], axis=(0, 2, 4))
        # Calculate dLdX
        dLdX = np.sum(dLdO[:, :, np.newaxis, np.newaxis, :, :] * W_reshaped[np.newaxis, :, :, :, :, :], axis=(1, 4, 5))
        # Store dLdW
        self.dLdW = dLdW
        return dLdX
