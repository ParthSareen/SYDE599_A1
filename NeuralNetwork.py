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
        if input_mat.shape[0] != output_mat.shape[0]:  # if number of input samples != number of output samples
            print("oh fuck")
            exit()
        if input_mat.shape[1] != self.layers[0].num_prev_nodes:  # if incorrect number of inputs
            print("oh fuck")
            exit()
        if output_mat.shape[1] != self.layers[-1].num_nodes:  # if incorrect number of outputs
            print("oh fuck")
            exit()

        losses = []
        for epoch in range(epochs):
            self.reset_gradients()
            samples_losses = []
            for x, y in zip(input_mat, output_mat):
                sample_loss_vec = self.single_pass(x, y)
                self.backward(sample_loss_vec)
                samples_losses.append(np.mean(sample_loss_vec))

            self.apply_gradient()

            losses.append(sum(samples_losses) / len(samples_losses))
        return losses

    def single_pass(self, input_vec: np.array, y_truth_vec: np.array) -> np.array:
        if input_vec.shape[0] != self.layers[0].num_prev_nodes:  # if incorrect number of inputs
            print("oh fuck")
            exit()
        if y_truth_vec.shape[0] != self.layers[-1].num_nodes:  # if incorrect number of outputs
            print("oh fuck")
            exit()

        y_pred = self.predict(input_vec)

        return self.loss(y_truth_vec, y_pred)

    def predict(self, input_vec: np.array) -> np.array:
        if input_vec.shape[0] != self.layers[0].num_prev_nodes:  # if incorrect number of inputs
            print("oh fuck")
            exit()

        x = input_vec
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def loss(self, y_truth_vec: np.array, y_pred_vec: np.array):
        if y_truth_vec.shape[0] != y_pred_vec[0]:  # if y and y_pred have different shapes
            print("oh fuck")
            exit()

        return np.power((y_pred_vec - y_truth_vec), 2) / 2

    def backward(self, y_truth: np.array):
        if y_truth.shape[0] != self.layers[-1].num_nodes:
            print("oh fuck")
            exit()

        delta = self.layers[-1].output_vec - y_truth
        for layer in self.layers:
            delta = layer.backward(delta)

    def apply_gradient(self):
        for layer in self.layers:
            layer.apply_grad(self.learning_rate)

    def reset_gradients(self):
        for layer in self.layers:
            layer.reset_grad()

    def display_grad(self, layer_num: int):
        print(f'Gradient of layer {layer_num}')
        self.layers[layer_num].print_grad()
