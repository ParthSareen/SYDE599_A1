import numpy as np

from Activation import Activation


class Layer:
    def __init__(self,
                 nodes: int,
                 prev_nodes: int,
                 activation_type: Activation=Activation.REGRESSION,
                 weights_mat: np.ndarray = np.array([]),
                 bias_vec: np.array = np.array([])):

        self.num_nodes = nodes
        self.num_prev_nodes = prev_nodes
        self.activation_type = activation_type

        if not np.any(weights_mat):  # if no values were given
            self.weights_mat = np.random.rand(nodes, prev_nodes)
        elif weights_mat.shape != (nodes, prev_nodes):  # if incorrect shape given
            raise "incorrect shape given"
        else:
            self.weights_mat = weights_mat

        if not np.any(bias_vec):  # if no values were given
            self.bias_vec = np.random.rand(nodes, )
        elif bias_vec.shape != (nodes, ):  # if incorrect shape given
            raise "incorrect shape given"
        else:
            self.bias_vec = bias_vec

        self.prev_nodes_vec = np.zeros((prev_nodes, ))
        self.activations_vec = np.zeros((nodes, ))
        self.output_vec = np.zeros((nodes, ))
        self.weights_grad_mat = np.zeros((nodes, prev_nodes))
        self.bias_grad_vec = np.zeros((nodes, ))
        self.delta_vec = np.zeros((nodes, ))

    def reset_grad(self):
        self.weights_grad_mat = np.zeros((self.num_nodes, self.num_prev_nodes))
        self.bias_grad_vec = np.zeros((self.num_nodes, ))
        self.delta_vec = np.zeros((self.num_nodes, ))

    def apply_grad(self, learning_rate: float, num_samples: int):
        self.weights_mat -= learning_rate * self.weights_grad_mat / num_samples
        self.bias_vec -= learning_rate * self.bias_grad_vec / num_samples

    def backward(self, next_grad: np.array):
        if next_grad.shape != (self.num_nodes, ):  # if incorrect shape given
            raise "incorrect shape given"

        if self.activation_type == Activation.RELU:
            self.delta_vec = np.where(self.activations_vec <= 0, 0, 1) * next_grad
        elif self.activation_type == Activation.REGRESSION:
            self.delta_vec = self.output_vec
        elif self.activation_type == Activation.SIGMOID:
            self.delta_vec = self.output_vec * (1 - self.output_vec) * next_grad
        else:
            raise "incorrect activation type"

        transposed_prev_nodes = np.transpose(np.expand_dims(self.prev_nodes_vec, axis=-1))
        w_grad = np.matmul(np.expand_dims(self.delta_vec, axis=-1), transposed_prev_nodes)
        b_grad = self.delta_vec
        self.add_grad(w_grad, b_grad)

        # return grad for the previous layer
        return np.matmul(np.transpose(self.weights_mat), self.delta_vec)

    def add_grad(self, w_grad: np.ndarray, b_grad: np.array):
        self.weights_grad_mat += w_grad
        self.bias_grad_vec += b_grad

    def print_grad(self):
        print(f'Gradient of weights: {self.weights_grad_mat}')
        print(f'Gradient of biases: {self.bias_grad_vec}')
        print('')

    def print_weights(self):
        print(f'Weights: {self.weights_mat}')
        print(f'Biases: {self.bias_vec}')
        print('')

    def forward(self, input_vec: np.array) -> np.array:
        if input_vec.shape != (self.num_prev_nodes, ):  # if incorrect shape given
            raise "incorrect shape given"

        self.prev_nodes_vec = input_vec
        self.activations_vec = np.matmul(self.weights_mat, input_vec) + self.bias_vec

        if self.activation_type == Activation.REGRESSION:
            self.output_vec = self.activations_vec
        elif self.activation_type == Activation.RELU:
            self.output_vec = np.maximum(0, self.activations_vec)
        elif self.activation_type == Activation.SIGMOID:
            self.output_vec = 1/(1 + np.exp(-self.activations_vec))
        else:
            raise "Incorrect activation type"

        return self.output_vec
