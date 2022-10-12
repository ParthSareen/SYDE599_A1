import numpy as np


class NeuralNetwork:
    def __init__(self, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate
        self.__setup__()

    def __setup__(self):
        # call methods with Layer class to set up the network initially
        pass

    def train(self, input_mat: np.ndarray, output_mat: np.ndarray, epochs: int):

        pass

    def single_pass(self, input_vec: np.ndarray, output_vec: np.ndarray):
        pass

    def predict(self, input_vec: np.array) -> np.ndarray:
        pass

    def loss(self, output_vec: np.array, output_vec_pred: np.array):
        pass

    def display_grad(self):
        pass
