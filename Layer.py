import numpy as np

from Activation import Activation


class Layer:
    def __init__(self,
                 nodes: int,
                 prev_nodes: int,
                 activation_type: Activation=Activation.NOTHING,
                 weights_mat: np.ndarray = np.array([]),
                 bias_vec: np.ndarray = np.array([])):
        pass

    def reset_grad(self):
        pass

    def apply_grad(self, learning_rate: float):
        pass

    def backward(self, next_delta: np.ndarray):
        pass

    def add_grad(self, w_grad: np.ndarray, b_grad: np.ndarray):
        pass

    def print_grad(self):
        pass

    def print_weights(self):
        pass

    def forward(self, input: np.ndarray):
        pass
