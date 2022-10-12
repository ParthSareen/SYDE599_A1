import numpy as np


class NeuralNetwork:
    def __init__(self, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate
        self.__setup__()

    def __setup__(self):
        # call methods with Layer class to set up the network initially
        pass

    def train(self, input_mat, output_mat, epochs):
        pass

    def single_pass(self, input_vec, output_vec):
        pass

    def predict(self, input_vec):
        pass

    def loss(self, output_vec, output_vec_pred):
        pass

    def display_grad(self):
        pass
