import numpy as np
from Layer import Layer


class NeuralNetwork:
    def __init__(self, layers: [Layer], learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.__setup__()

    def __setup__(self):
        # call methods with Layer class to set up the network initially
        pass

    def train(self, input_mat: np.ndarray, output_mat: np.ndarray, epochs: int) -> list[float]:
        """
        Trains the network using input_mat and output_mat for number of epochs
        :param input_mat: [#samples, input_size] np array
        :param output_mat: [#samples, output_Size] np array
        :param epochs: int
        :return: list of losses (one per epoch)
        """
        if input_mat.shape[0] != output_mat.shape[0]:  # if number of input samples != number of output samples
            raise "input and output shape differ"
        if input_mat.shape[1] != self.layers[0].num_prev_nodes:  # if incorrect number of inputs
            raise "incorrect input shape given"
        if output_mat.shape[1] != self.layers[-1].num_nodes:  # if incorrect number of outputs
            raise "incorrect output shape given"

        losses = []
        for epoch in range(epochs):
            self.reset_gradients()
            samples_losses = []
            for x, y in zip(input_mat, output_mat):
                sample_loss_vec = self.single_pass(x, y)

                self.backward(y)
                samples_losses.append(np.mean(sample_loss_vec))

            print("done epoch, applying grad")
            self.apply_gradient(input_mat.shape[0])

            losses.append(sum(samples_losses) / len(samples_losses))

        print(losses)
        return losses

    def single_pass(self, input_vec: np.array, y_truth_vec: np.array) -> np.array:
        if input_vec.shape[0] != self.layers[0].num_prev_nodes:  # if incorrect number of inputs
            raise "incorrect input shape given"
        if y_truth_vec.shape[0] != self.layers[-1].num_nodes:  # if incorrect number of outputs
            raise "incorrect output shape given"

        y_pred = self.predict(input_vec)

        return self.loss(y_truth_vec, y_pred)

    def predict(self, input_vec: np.array) -> np.array:
        if input_vec.shape[0] != self.layers[0].num_prev_nodes:  # if incorrect number of inputs
            raise "incorrect input shape given"

        x = input_vec
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def loss(self, y_truth_vec: np.array, y_pred_vec: np.array):
        if y_truth_vec.shape[0] != y_pred_vec.shape[0]:  # if y and y_pred have different shapes
            raise "y_truth and y_pred must have same shape"

        return np.power((y_pred_vec - y_truth_vec), 2) / 2

    def backward(self, y_truth: np.array):
        if y_truth.shape[0] != self.layers[-1].num_nodes:
            raise "incorrect y_truth shape given"

        delta = self.layers[-1].output_vec - y_truth

        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def apply_gradient(self, num_samples: int):
        for layer in self.layers:
            layer.apply_grad(self.learning_rate, num_samples)

    def reset_gradients(self):
        for layer in self.layers:
            layer.reset_grad()

    def display_grad(self, layer_num: int):
        print(f'Gradient of layer {layer_num}')
        self.layers[layer_num].print_grad()
