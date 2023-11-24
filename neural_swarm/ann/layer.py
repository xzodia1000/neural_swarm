import numpy as np


class Layer:
    def __init__(
        self,
        nodes,
        activation=None,
        previous_layer=None,
        next_layer=None,
    ):
        """Initializes the layer with given parameters.

        Args:
            nodes (int): Number of nodes (neurons) in this layer.
            activation (Activation, optional): Activation function for this layer.
            previous_layer (Layer, optional): Reference to the previous layer in the network.
            next_layer (Layer, optional): Reference to the next layer in the network.
        """
        self.nodes = nodes
        self.activation = activation
        self.previous_layer = previous_layer
        self.next_layer = next_layer
        self.weighted_sum = None  # Stores the weighted sum (input * weights)
        self.input = None  # Stores the input to the layer
        self.weights = None  # Weights matrix for the layer
        self.bias = None  # Bias vector for the layer
        self.delta = None  # Used in backpropagation

    def forward(self, input):
        """Performs the forward pass through the layer.

        Args:
            input (ndarray): The input data or output from the previous layer.

        Returns:
            ndarray: The output of the layer after applying the activation function.
        """
        self.input = input
        # Computes the weighted sum and applies the activation function
        self.weighted_sum = (
            self.activation.evaluate(np.dot(self.input, self.weights))
            if self.next_layer is not None
            else self.activation.evaluate(self.input)
        )
        return self.weighted_sum

    def set_weights(self, weights=None):
        """Sets the weights of the layer.

        Args:
            weights (ndarray, optional): A numpy array of weights to be set.
                                         If None, initialize using `initialize_weights`.

        Raises:
            ValueError: If the shape of provided weights doesn't match the expected shape.
        """
        if self.weights is None:
            self.weights = self.initialize_weights()
        elif weights.shape != self.weights.shape:
            raise ValueError("Weights shape mismatch")
        elif weights.shape == self.weights.shape:
            self.weights = weights

    def set_activation(self, activation):
        """Sets the activation function for the layer.

        Args:
            activation (Activation): The activation function to use.
        """
        self.activation = activation

    def initialize_weights(self):
        """Initializes weights for the layer based on the number of nodes in this and the next layer.

        Returns:
            ndarray: Initialized weight matrix.
        """
        # Heuristic for setting the initial weights
        limit = np.sqrt(6.0 / (self.nodes + self.next_layer.nodes))
        return np.random.uniform(
            -limit, limit, size=(self.nodes, self.next_layer.nodes)
        )

    def initialize_bias(self):
        """Initializes the bias for the layer.

        Returns:
            ndarray: Initialized bias vector.
        """
        return np.zeros((self.input.shape[0], self.next_layer.nodes))

    def get_input(self):
        """Returns the input received by the layer.

        Returns:
            ndarray: The input to the layer.
        """
        return self.input

    def update_input(self, input):
        """Updates the input of the layer.

        Args:
            input (ndarray): The new input to set for the layer.
        """
        self.input = input
