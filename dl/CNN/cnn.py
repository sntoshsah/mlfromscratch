import numpy as np
# from datasetup import X_data, y_data, one_hot

class ConvLayer:
    def __init__(self, kernel_size):
        # Initialize a random 3x3 kernel for a toy example
        self.kernel = np.random.randn(kernel_size, kernel_size) * 0.1
        self.bias = np.random.randn(1) * 0.1

    def forward(self, input_data):
        self.input = input_data
        in_h, in_w = input_data.shape
        k_h, k_w = self.kernel.shape

        # Output dimensions
        out_h, out_w = in_h - k_h + 1, in_w - k_w + 1
        output = np.zeros((out_h, out_w))

        # Perform valid convolution
        for i in range(out_h):
            for j in range(out_w):
                output[i, j] = np.sum(input_data[i:i+k_h, j:j+k_w] * self.kernel) + self.bias.item()
        return output

    def backward(self, output_gradient, learning_rate):
        """
        output_gradient (dL/dY) has the shape of the output from forward pass.
        """
        k_h, k_w = self.kernel.shape
        grad_weights = np.zeros((k_h, k_w))
        grad_input = np.zeros(self.input.shape)

        # 1. Gradient with respect to Weights (dL/dW)
        # We convolve the input with the output gradient
        for i in range(k_h):
            for j in range(k_w):
                # Region of input that affected this specific weight
                input_region = self.input[i : i + output_gradient.shape[0],
                                          j : j + output_gradient.shape[1]]
                grad_weights[i, j] = np.sum(input_region * output_gradient)

        # 2. Gradient with respect to Bias (dL/db)
        grad_bias = np.sum(output_gradient)

        # 3. Gradient with respect to Input (dL/dX)
        # To pass the error to the previous layer, we convolve output_gradient
        # with the flipped kernel (Full Convolution)
        flipped_kernel = np.flip(self.kernel)
        # Padding output_gradient to allow "Full" convolution
        padded_grad = np.pad(output_gradient, k_h - 1, mode='constant')

        for i in range(self.input.shape[0]):
            for j in range(self.input.shape[1]):
                grad_input[i, j] = np.sum(padded_grad[i:i+k_h, j:j+k_w] * flipped_kernel)

        # Update Weights and Bias
        self.kernel -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias.item()

        return grad_input
    

class FlattenLayer:
    def forward(self, input_data):
        self.input_shape = input_data.shape
        return input_data.flatten().reshape(1, -1)

    def backward(self, output_gradient):
        # Reshape the 1D gradient back to the 2D (or 3D) spatial shape
        return output_gradient.reshape(self.input_shape)
    
def softmax(x):
    exps = np.exp(x - np.max(x)) # Subtract max for numerical stability
    return exps / np.sum(exps)

def cross_entropy_loss(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-9))
