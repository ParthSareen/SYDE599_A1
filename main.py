import pickle 
import numpy
import numpy as np
import matplotlib.pyplot as plt

from Activation import Activation
from NeuralNetwork import *
from Layer import *


def plot(losses):
    plt.figure()
    plt.scatter(range(len(losses)), losses)
    plt.title("Model training losses")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()


def import_data():
    with open('datafiles/assignment-one-test-parameters.pkl', 'rb') as f:
        data = pickle.load(f)

    weights_layer_one = data['w1']
    weights_layer_two = data['w2']
    weights_layer_three = data['w3']
    bias_layer_one = data['b1']
    bias_layer_two = data['b2']
    bias_layer_three = data['b3']
    inputs = data['inputs']
    targets = data['targets']

    parameters = [
        (weights_layer_one, bias_layer_one),
        (weights_layer_two, bias_layer_two),
        (weights_layer_three, bias_layer_three)
    ]
    training_data = (inputs, targets)

    return training_data, parameters


def create_network():
    layer_one = Layer(10, 2, activation_type=Activation.RELU)
    layer_two = Layer(10, 10, activation_type=Activation.RELU)
    layer_three = Layer(1, 10, activation_type=Activation.REGRESSION)

    layers = [layer_one, layer_two, layer_three]
    return NeuralNetwork(layers)


def main():
    training_data, parameters = import_data()

    network = create_network()
    network.set_weights(parameters)

    inputs_array = np.array(training_data[0])
    targets_array = np.expand_dims(np.array(training_data[1]), -1)

    # to print gradients of untrained network
    network.reset_gradients()
    network.single_pass(inputs_array[0, :], targets_array[0, :])
    network.backward(targets_array[0, :])
    network.display_grad(layer_num=0)

    # train the network and plot the losses
    losses = network.train(inputs_array, targets_array, epochs=5)
    # append the loss after the last epoch by evaluating on the dataset
    losses.append(network.evaluate(inputs_array, targets_array))
    plot(losses)


if __name__ == '__main__': 
    main()
