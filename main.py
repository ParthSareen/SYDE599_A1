import pickle 
import numpy
from Activation import Activation
from NeuralNetwork import *
from Layer import *


def main(): 
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
    layer_one = Layer(10, 2, Activation.RELU, weights_layer_one, bias_layer_one)
    layer_two = Layer(10, 10, Activation.RELU, weights_layer_two, bias_layer_two)
    layer_three = Layer(1, 10, Activation.REGRESSION, weights_layer_three, bias_layer_three)

    layers = [layer_one, layer_two, layer_three]

    network = NeuralNetwork(layers)

    # for sample in inputs:
    #     print(network.predict(sample))

    network.train(np.array(inputs), np.expand_dims(np.array(targets), -1), epochs=5)
    # pred = network.predict(inputs[0])
    # print(pred)
    for sample in inputs:
        print(network.predict(sample))


if __name__ == '__main__': 
    main()
