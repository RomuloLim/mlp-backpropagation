# MLP Backpropagation
This repository contains the implementation of a Multi-Layer Perceptron (MLP) neural network using the backpropagation algorithm.

## Overview
The neural network implemented in this project consists of multiple fully connected layers and uses backpropagation to adjust the weights during training. The activation function can be either sigmoid or tanh.

## Features
- Initialize the network with random weights.
- Forward propagate inputs through the network.
- Calculate neuron activation.
- Backward propagate errors through the network.
- Update network weights.
- Predict classes for new data.
- Evaluate model accuracy using cross-validation.
- Visualize the neural network and training error.

## Dependencies
- Python 3.x
- Matplotlib
- NetworkX

## How to Use
1. Clone this repository:
```sh
git clone https://github.com/RomuloLim/mlp-backpropagation
```

2. Install dependencies:
```sh
pip install matplotlib networkx
```

3. Run the main script:
```sh
python main.py
```

## Code Structure
- initialize_network(n_inputs, n_hidden, n_outputs): Initializes the neural network.
- forward_propagate(network, row): Performs forward propagation.
- backward_propagate_error(network, expected): Performs backpropagation of errors.
- update_weights(network, row, l_rate): Updates the network weights.
- train_network(network, train, learning_rate, epochs, n_outputs): Trains the neural network.
- predict(network, row): Makes predictions with the trained network.
- draw_network(network): Draws the neural network structure.
- load_csv(filename): Loads a CSV file with data.

## Example Usage
An example usage is classifying Iris flowers using the `iris.csv` dataset. The code loads the dataset, normalizes the data, trains the neural network, and evaluates the model's accuracy.

## Results
After training, the neural network generates accuracy results displayed in the console and plots the errors per training epoch.
