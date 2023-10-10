import numpy as np


class Layer:
    def __init__(
        self,
        nodes,
        activation=None,
        previous_layer=None,
        next_layer=None,
        input=None,
        y_true=None,
    ):
        self.nodes = nodes
        self.activation = activation
        self.previous_layer = previous_layer
        self.next_layer = next_layer
        self.weighted_sum = None
        self.input = input
        self.y_true = y_true
        self.weights = None
        self.bias = None
        self.delta = None

    def forward(self):
        if self.weights is None and self.input is None:
            self.weights = self.initialize_weights()
            self.bias = self.initialize_bias()

        self.weighted_sum = (
            self.activation.evaluate(
                np.matmul(self.previous_layer.weighted_sum, self.weights.T) + self.bias
            )
            if self.input is None
            else self.input
        )

        return self.weighted_sum

    def backward(self, learning_rate):
        self.calculate_delta()

        if self.weights is not None:
            self.update(learning_rate)

        return self.delta

    def calculate_delta(self):
        if self.y_true is not None:
            true = np.array(
                self.y_true,
            ).reshape(1, self.y_true.shape[0])
            weighted_sum = self.weighted_sum.T

            error = true - weighted_sum
            self.delta = self.activation.derivative(weighted_sum) * error

        elif self.activation is not None:
            self.delta = self.activation.derivative(self.weighted_sum.T) * (
                np.dot(self.next_layer.weights.T, self.next_layer.delta)
            )

    def update(self, learning_rate):
        self.weights -= learning_rate * np.matmul(
            self.delta, self.previous_layer.weighted_sum
        )
        # self.bias -= learning_rate * self.delta

    def initialize_weights(self):
        limit = np.sqrt(6.0 / (self.nodes + self.previous_layer.nodes))
        return np.random.uniform(
            -limit, limit, size=(self.nodes, self.previous_layer.nodes)
        )

    def initialize_bias(self):
        return np.zeros(self.nodes)
