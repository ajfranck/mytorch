import numpy as np
import torch
# you may import other functions/files from mytorch

class RNN:

    def __init__(self, num_inputs, num_hiddens):
        """
        Initialize the weights to be zero-mean Gaussian with 
        variance 0.01. Ignore the bias term.

        :param num_inputs: Dimension of inputs.
        :param num_hiddens: Dimension of hidden state.
        """
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.Wxh = (torch.randn(num_inputs, num_hiddens) * 0.01)
        self.Whh = (torch.randn(num_hiddens, num_hiddens) * 0.01)
        self.params = [self.Wxh, self.Whh]


    def forward(self, inputs, state=None):
        """
        Forward operation of RNN layer. Performs
        operation H_t+1 = tanh(Xt*Wxh + Ht*Whh).

        :param inputs: Input data matrix with shape (num_steps, batch_size, num_inputs).
        :param state: Initial hidden state with shape (batch_size, num_hiddens).
        :return outputs: Output data matrix after linear transformation.
        :return state: Final hidden state for each element in the batch.
        """
        if state is None:
            state = np.zeros((inputs.shape[1], self.num_hiddens))

        outputs = []
        for X in inputs:
            part1 = X @ self.Wxh
            part2 = state @ self.Whh
            state = torch.tanh(torch.from_numpy(part1 + part2)).numpy()
            outputs.append(state)
        
        outputs = np.array(outputs)
        return outputs, state

    def backward(self, dLdO):
            """
            Backpropagation operation for variables in RNN
            layer. Stores derivatives dLdWxh, dLdWhh.

            :param dLdO: Derivative of loss with respect to output.
            Obtained from backward operation on loss object.
            :returns None:
            """

            dLd0 = dLdO
            dLdWxh = np.zeros_like(self.Wxh)
            dLdWhh = np.zeros_like(self.Whh)
            for i in range(len(self.inputs)):
                dLdWxh += self.inputs[i].T @ dLd0[i]
                dLdWhh += self.states[i].T @ dLd0[i]
                dLd0[i] = dLd0[i] @ self.W_hh.T
                dLd0[i] = dLd0[i] * (1 - self.states[i] ** 2)
            self.dLdWxh = dLdWxh
            self.dLdWhh = dLdWhh
            
            return None
            