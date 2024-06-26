from random import seed
from random import random
import matplotlib.pyplot as plt
import networkx as nx
import math
import pandas as pd


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


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    errors = []
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        errors.append(sum_error)
    return errors


def draw_network(network):
    G = nx.DiGraph()
    for i, layer in enumerate(network):
        for j, neuron in enumerate(layer):
            for k, weight in enumerate(neuron['weights']):
                from_node = (i - 1, k) if i > 0 else k
                to_node = (i, j)
                G.add_node(from_node, layer=i - 1 if i > 0 else 'input')
                G.add_node(to_node, layer=i)
                G.add_edge(from_node, to_node, weight=weight)

    pos = nx.multipartite_layout(G, subset_key="layer", scale=2)
    # o valor escrito nas linhas está sobrepondo outros, correção:
    for key, value in pos.items():
        pos[key] = (value[0], value[1] + 0.1)

    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=10, font_weight='bold',
            font_color='black', edge_color='gray', width=1.5, alpha=0.9, arrowsize=10)

    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


def use_dataset():
    # Flowers mapping
    flowers = {
        'Setosa': 0,
        'Versicolor': 1,
        'Virginica': 2
    }

    csv_data = pd.read_csv('iris.csv').values

    for data in csv_data:
        data[-1] = flowers[data[-1]]

    return csv_data


seed(1)

transfer_neuron_function = 'sigmoid'
dataset = use_dataset()
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))

network = initialize_network(n_inputs, 3, n_outputs)

# Train the network and get the errors
errors = train_network(network, dataset, 0.024, 3000, n_outputs)

# Plot the errors
plt.plot(errors)
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Aprendizado da rede neural')
plt.show()

draw_network(network)

success = 0
errors = 1

for row in dataset:
    prediction = predict(network, row)
    if prediction == row[-1]:
        success += 1
    else:
        errors += 1
    print('Expected=%d, Got=%d' % (row[-1], prediction))

print('Success: ', success)
print('Errors: ', errors)

# for layer in network:
#     print(layer)
