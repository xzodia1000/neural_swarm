class Network:
    def __init__(self):
        """Initializes the neural network with an empty list of layers."""
        self.layers = []  # Stores the layers of the network

    def add_layer(self, layer):
        """Adds a layer to the network and sets up connections with adjacent layers.

        Args:
            layer (Layer): The layer to be added to the network.
        """
        self.layers.append(layer)  # Add the new layer to the list

        # If there are at least two layers, set up connections between the last two layers
        if len(self.layers) >= 2:
            # Setting the previous layer for the newly added layer
            self.layers[-1].previous_layer = self.layers[-2]
            # Setting the next layer for the second-last layer
            self.layers[-2].next_layer = self.layers[-1]
            # Initialize weights for the second-last layer
            self.layers[-2].set_weights()

    def forward(self, input):
        """Performs a forward pass through the network.

        Args:
            input (ndarray): The input data for the network.

        Returns:
            ndarray: The output from the last layer after processing the input.
        """
        output = input
        for layer in self.layers:  # Iterating through each layer
            output = layer.forward(output)  # Forward pass through the current layer

        return output  # Return the final output after passing through all layers
