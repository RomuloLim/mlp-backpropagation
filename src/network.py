from random import seed
from random import random
import math


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation using sigmoid function
def transfer_sigmoid(activation):
    return 1.0 / (1.0 + math.exp(-activation))


# Transfer neuron activation using tanh function
def transfer_tanh(activation):
    return math.tan(activation)


# Transfer neuron activation
def transfer(activation):
    match transfer_neuron_function:
        case 'sigmoid':
            return transfer_sigmoid(activation)
        case 'tanh':
            return transfer_tanh(activation)
        case _:
            return transfer_sigmoid(activation)


# Derivative of the sigmoid function
def transfer_derivative_sigmoid(output):
    return output * (1.0 - output)


# Derivative of the tanh function
def transfer_derivative_tanh(output):
    return 1.0 - output ** 2


# Derivative of the transfer function
def transfer_derivative(output):
    match transfer_neuron_function:
        case 'sigmoid':
            return transfer_derivative_sigmoid(output)
        case 'tanh':
            return transfer_derivative_tanh(output)
        case _:
            return transfer_derivative_sigmoid(output)


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row

    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs

    return inputs


seed(1)
transfer_neuron_function = 'sigmoid'
network = initialize_network(2, 1, 2)

print('=== LAYERS ===')
for layer in network:
    print(layer)
print('==============')

print('=== FORWARD PROPAGATE ===')
row = [1, 0, None]
output = forward_propagate(network, row)
print(output)
