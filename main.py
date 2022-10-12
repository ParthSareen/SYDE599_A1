import pickle 
import numpy
from Activation import Activation
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
    layer_one = Layer(len(weights_layer_one), 0, Activation.NOTHING, weights_layer_one, bias_layer_one)
    layer_two = Layer(len(weights_layer_two), len(weights_layer_one), Activation.NOTHING, weights_layer_two, bias_layer_two)
    layer_three = Layer(len(weights_layer_three), len(weights_layer_two), Activation.NOTHING, weights_layer_three, bias_layer_three)
    



if __name__ == '__main__': 
    main()