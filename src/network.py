from random import seed
from random import random
from random import randrange
import matplotlib.pyplot as plt
import networkx as nx
import math
from csv import reader


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
def train_network(network, train, learning_rate, epochs, n_outputs):
    errors = []
    for epoch in range(epochs):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, learning_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
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

    pos = nx.multipartite_layout(G, subset_key="layer", scale=10)

    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=10, font_weight='bold',
            font_color='black', edge_color='gray', width=1.5, alpha=0.9, arrowsize=10)

    labels = nx.get_edge_attributes(G, 'weight')

    labels = {k: round(v, 4) for k, v in labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8, font_weight='bold', alpha=0.9, label_pos=0.2, rotate=False)
    plt.show()


def load_csv(filename):
    # Flowers mapping
    flowers = {
        'Setosa': 0,
        'Versicolor': 1,
        'Virginica': 2
    }

    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row or row[0] == 'sepal.length':
                continue
            row[-1] = flowers[row[-1]]
            print(row[0])
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    errors = train_network(network, train, l_rate, n_epoch, n_outputs)

    plt.plot(errors)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.show()

    draw_network(network)

    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions


seed(1)

transfer_neuron_function = 'sigmoid'
dataset = load_csv('iris.csv')

for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)

# convert class column to integers
str_column_to_int(dataset, len(dataset[0]) - 1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 3
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Precisão: %.3f%%' % (sum(scores) / float(len(scores))))
