import numpy as np

class Conv2d:
    
    def __init__(self, in_channels, out_channels, kernel_size):
        """
        Initialize the weights to be zero-mean Gaussian. You do not need to include
        a bias term.

        :param num_inputs: Number of inputs to layer.
        :param num_outputs: Number of outputs after layer.
        """
        self.W = np.random.randn(out_channels, in_channels, kernel_size[0], kernel_size[1])*0.1
        
    def corr2d(self, X, K):
        """Compute 2D cross-correlation. Taken from d2l.ai."""
        h, w = K.shape
        Y = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
        return Y

    def corr2d_multi_in(self, X, K):
        """Compute 2D cross-correlation with multiple input channels. Taken from d2l.ai."""
        # Iterate through the 0th dimension (channel) of K first, then add them up
        return sum(self.corr2d(x, k) for x, k in zip(X, K))
    
    def corr2d_multi_in_out(self, X, K):
        """Compute 2D cross-correlation with multiple input and output channels. Taken from d2l.ai."""
        # Iterate through the 0th dimension of K, and each time, perform
        # cross-correlation operations with input X. All of the results are
        # stacked together
        return np.stack([self.corr2d_multi_in(X, k) for k in K], 0)

    def forward(self, X):
        """
        Forward operation of convolutional layer.

        :param X: Input image batch of size (batch_size, in_channels, height, width).
        :return O: Output feature map.
        """
        self.X = X
        O = np.stack([self.corr2d_multi_in_out(X[ii, :, :, :], self.W) 
                         for ii in range(X.shape[0])], 0)
        return O
    
    def backward(self, dLdO):
        """
        Backward operation of convolutional layer. Stores derivative dLdW, and returns
        dLdX.

        :param dLdO: Derivative of loss with respect to output.
        Obtained from backward operation on loss object.
        :returns dLdX: Derivative of loss with respect to input.
        """
        dLdW = np.zeros(self.W.shape)
        for c in range(self.W.shape[1]):
            for d in range(self.W.shape[0]):
                dLdW[d, c, :, :] = sum((self.corr2d(x, k) for x, k in zip(self.X[:,c,:,:], dLdO[:,d,:,:])), 0)

        kernel_size = dLdW.shape[-2:]
        in_channels = dLdW.shape[1]
        batch_size = self.X.shape[0]
        pad_height = kernel_size[0] - 1
        pad_width = kernel_size[1] - 1
        pad_size = ((0, 0), (pad_height, pad_width), (pad_height, pad_width))
        dLdX = np.zeros(self.X.shape)
        for cc in range(in_channels):
            fW = np.flip(np.flip(self.W[:, cc, :, :], 1), 2)
            dLdX[:, cc, :, :] = np.stack([self.corr2d_multi_in(np.pad(dLdO[ii, :, :, :], pad_size), fW) 
                                          for ii in range(batch_size)], 0) 
        
        self.dLdW = dLdW
        return dLdX