import numpy as np


class Layer:
    def __init__(
        self,
        nodes,
        activation=None,
        previous_layer=None,
        next_layer=None,
    ):
        self.nodes = nodes
        self.activation = activation
        self.previous_layer = previous_layer
        self.next_layer = next_layer
        self.weighted_sum = None
        self.input = None
        self.weights = None
        self.bias = None
        self.delta = None

    def forward(self, input):
        self.input = input
        self.weighted_sum = (
            self.activation.evaluate(np.dot(self.input, self.weights))
            if self.next_layer is not None
            else self.activation.evaluate(self.input)
        )
        return self.weighted_sum

    def set_weights(self, weights=None):
        if self.weights is None:
            self.weights = self.initialize_weights()
        elif weights.shape != self.weights.shape:
            raise ValueError("Weights shape mismatch")
        elif weights.shape == self.weights.shape:
            self.weights = weights

    def set_activation(self, activation):
        self.activation = activation

    def initialize_weights(self):
        limit = np.sqrt(6.0 / (self.nodes + self.next_layer.nodes))
        return np.random.uniform(
            -limit, limit, size=(self.nodes, self.next_layer.nodes)
        )

    def initialize_bias(self):
        return np.zeros((self.input.shape[0], self.next_layer.nodes))

    def get_input(self):
        return self.input

    def update_input(self, input):
        self.input = input
