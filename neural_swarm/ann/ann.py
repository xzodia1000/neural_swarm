import numpy as np
from neural_swarm.ann.layer import Layer
from neural_swarm.ann.network import Network


class ANN:
    def __init__(self):
        """Initializes an artificial neural network with an empty network and dataset."""
        self.network = Network()  # Initialize a new network
        self.dataset = None  # Placeholder for dataset

    def add_layer(self, activation, nodes):
        """Adds a layer to the neural network.

        Args:
            activation (Activation): The activation function for the layer.
            nodes (int): The number of nodes in the layer.
        """
        layer = Layer(activation=activation, nodes=nodes)  # Create a new layer
        self.network.add_layer(layer)  # Add the layer to the network

    def run(self, x):
        """Runs the neural network on a given input.

        Args:
            x (ndarray): The input data.

        Returns:
            ndarray: The output of the network for the given input.
        """
        return self.network.forward(x)  # Perform a forward pass through the network

    def evaluate(self, x, y, loss):
        """Evaluates the neural network on given data and loss function.

        Args:
            x (ndarray): The input data.
            y (ndarray): The true labels.
            loss (Loss): The loss function to use for evaluation.

        Returns:
            tuple: A tuple containing the accuracy and the loss value.
        """
        y = np.array(y)
        y_pred = self.network.forward(x)  # Forward pass to get predictions

        try:
            y_pred = np.array(y_pred).reshape(
                y.shape
            )  # Reshape predictions to match y's shape
        except:
            pass  # If reshape fails, continue without altering y_pred

        ann_loss = loss.evaluate(y_pred, y)  # Compute the loss

        accuracy = 0  # Initialize accuracy
        # Different accuracy calculations based on the type of loss function
        if loss.__str__() == "BinaryCrossEntropy":
            y_pred = (y_pred >= 0.5).astype(int)  # Convert predictions to binary
            accuracy = np.mean(y_pred == y)  # Calculate binary classification accuracy
            ann_loss = loss.evaluate(
                y_pred, y
            )  # Recalculate loss for binary predictions
        elif loss.__str__() == "CategoricalCrossEntropy":
            accuracy = self.categorical_classification_accuracy(
                y, y_pred
            )  # Calculate multi-class accuracy
        elif loss.__str__() == "Mse":
            accuracy = np.mean(
                (y - y_pred) ** 2
            )  # Calculate mean squared error as accuracy

        return accuracy * 100, ann_loss  # Return accuracy (percentage) and loss

    def categorical_classification_accuracy(self, y_true, y_pred):
        """Calculates accuracy for categorical classification.

        Args:
            y_true (ndarray): True labels.
            y_pred (ndarray): Predicted labels.

        Returns:
            float: The classification accuracy.
        """
        y_true = np.argmax(
            y_true, axis=1
        )  # Convert one-hot encoded labels to class indices
        y_pred = np.argmax(y_pred, axis=1)  # Convert predictions to class indices
        return np.mean(y_pred == y_true)  # Calculate the mean accuracy
