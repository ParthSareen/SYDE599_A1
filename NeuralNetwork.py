import numpy as np
from Layer import *


class NeuralNetwork:
    def __init__(self, layers=None, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate

    def train(self, input_mat: np.ndarray, output_mat: np.ndarray, epochs: int):

        pass

    def single_pass(self, input_vec: np.ndarray, output_vec: np.ndarray):
        output_vec_pred = self.predict(input_vec)
        loss_vec = self.loss(output_vec, output_vec_pred)
        return loss_vec

    def predict(self, input_vec: np.array) -> np.ndarray:
        pass

    def loss(self, output_vec: np.array, output_vec_pred: np.array) -> np.ndarray:
        pass

    def display_grad(self):
        pass