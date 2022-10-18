import numpy as np

from Activation import Activation


class Layer:
    def __init__(self,
                 nodes: int,
                 prev_nodes: int,
                 activation_type: Activation=Activation.REGRESSION,
                 weights_mat: np.ndarray = np.array([]),
                 bias_vec: np.ndarray = np.array([])):

        self.num_nodes = nodes
        self.num_prev_nodes = prev_nodes
        self.activation_type = activation_type

        if not np.any(weights_mat):  # if no values were given
            self.weights_mat = np.random.rand(prev_nodes, nodes)
        elif weights_mat.shape != (prev_nodes, nodes):  # if incorrect shape given
            print("oh fuck")
            exit()
        else:
            self.weights_mat = weights_mat

        if not np.any(bias_vec):  # if no values were given
            self.bias_vec = np.random.rand(nodes, 1)
        elif bias_vec.shape != (nodes, 1):  # if incorrect shape given
            print("oh fuck")
            exit()
        else:
            self.bias_vec = bias_vec

        self.prev_nodes_vec = np.zeros((prev_nodes, 1))
        self.activations_vec = np.zeros((nodes, 1))
        self.output_vec = np.zeros((nodes, 1))
        self.weights_grad_mat = np.zeros((prev_nodes, nodes))
        self.bias_grad_vec = np.zeros((nodes, 1))
        self.delta_vec = np.zeros((nodes, 1))

    def reset_grad(self):
        self.weights_grad_mat = np.zeros((self.num_prev_nodes, self.num_nodes))
        self.bias_grad_vec = np.zeros((self.num_nodes, 1))
        self.delta_vec = np.zeros((self.num_nodes, 1))

    def apply_grad(self, learning_rate: float):
        self.weights_mat -= learning_rate * self.weights_grad_mat
        self.bias_vec -= learning_rate * self.bias_grad_vec

    def backward(self, next_grad: np.ndarray):
        if next_grad.shape != (self.num_nodes, 1):  # if incorrect shape given
            print("oh fuck")
            exit()

        if self.activation_type == Activation.RELU:
            self.delta_vec = np.where(self.activations_vec <= 0, 0, 1) * next_grad
        elif self.activation_type == Activation.REGRESSION:
            self.delta_vec = self.output_vec - next_grad
        elif self.activation_type == Activation.SIGMOID:
            self.delta_vec = self.output_vec * (1 - self.output_vec) * next_grad
        else:
            print("oh fuck")
            exit()

        self.add_grad(self.delta_vec * self.prev_nodes_vec, self.delta_vec)

        # return grad for the previous layer
        return np.matmul(self.weights_mat, self.delta_vec)

    def add_grad(self, w_grad: np.ndarray, b_grad: np.ndarray):
        if w_grad.shape == (self.num_prev_nodes, self.num_nodes):
            self.weights_mat += w_grad
        else:  # if incorrect shape given
            print("oh fuck")
            exit()

        if b_grad.shape == (self.num_nodes, 1):
            self.bias_grad_vec += b_grad
        else:  # if incorrect shape given
            print("oh fuck")
            exit()

    def print_grad(self):
        print(f'Gradient of weights: {self.weights_grad_mat}')
        print(f'Gradient of biases: {self.bias_grad_vec}')
        print('')

    def print_weights(self):
        print(f'Weights: {self.weights_mat}')
        print(f'Biases: {self.bias_vec}')
        print('')

    def forward(self, input_vec: np.ndarray) -> np.ndarray:
        if input_vec.shape != (self.num_prev_nodes, 1):  # if incorrect shape given
            print("oh fuck")
            exit()

        self.prev_nodes_vec = input_vec
        self.activations_vec = np.matmul(np.transpose(self.weights_mat), input_vec) + self.bias_vec

        if self.activation_type == Activation.REGRESSION:
            self.output_vec = self.activations_vec
        elif self.activation_type == Activation.RELU:
            self.output_vec = np.maximum(0, self.activations_vec)
        elif self.activation_type == Activation.SIGMOID:
            self.output_vec = 1/(1 + np.exp(-self.activations_vec))
        else:
            print("oh fuck")
            exit()

        return self.output_vec
