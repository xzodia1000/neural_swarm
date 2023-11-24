import numpy as np
from neural_swarm.ann.activation import Identity, Sigmoid, Relu, Softmax, Tanh
from neural_swarm.pso.fitness_function import FitnessFunction


class ANNFunction(FitnessFunction):
    def __init__(self, ann, x, y, loss, optimize_activation=False):
        """Initializes the ANN function for fitness evaluation in particle swarm optimization.

        Args:
            ann: The artificial neural network to be optimized.
            x: Input data for the ANN.
            y: True labels corresponding to the input data.
            loss: The loss function used to evaluate the ANN.
            optimize_activation (bool, optional): Flag to optimize activation functions. Defaults to False.
        """
        self.ann = ann
        self.x = x
        self.y = y
        self.loss = loss
        self.optimize_activation = optimize_activation

        # Mapping integer indices to different activation functions
        self._activation_functions_map = {
            0: Sigmoid(),
            1: Relu(),
            2: Tanh(),
            3: Softmax(),
            4: Identity(),
        }

    def get_ann(self):
        """Returns the ANN instance.

        Returns:
            The artificial neural network instance.
        """
        return self.ann

    def get_dimension(self):
        """Calculates the dimension of the particle based on the ANN's weights and optionally activation functions.

        Returns:
            int: The dimension of the particle.
        """
        dimension = 0
        for layer in self.ann.network.layers:
            if layer.weights is not None:
                # Dimension is the number of weights and optionally an activation function index
                dimension += (
                    np.array(layer.weights).flatten().shape[0] + 1
                    if self.optimize_activation
                    else np.array(layer.weights).flatten().shape[0]
                )

        return dimension

    def set_variable(self, particle):
        """Decodes the particle into the ANN's weights and optionally activation functions.

        Args:
            particle: The encoded particle representing ANN's parameters.

        Returns:
            List: Decoded weights and optionally activation functions.
        """
        decoded = []
        encoded = np.array(particle)
        for layer in self.ann.network.layers:
            if layer.weights is not None:
                # Set weights for each layer
                weights = encoded[: layer.weights.size].reshape(layer.weights.shape)
                layer.set_weights(weights)
                decoded.append(weights)
                encoded = encoded[layer.weights.size :]

                # Set activation function if optimization is enabled
                if self.optimize_activation:
                    activation_index = int(np.clip(np.floor(encoded[0] * 5), 0, 4))
                    activation_function = self._activation_functions_map[
                        activation_index
                    ]
                    layer.set_activation(activation_function)
                    decoded.append(activation_function)
                    encoded = encoded[1:]

        return decoded

    def evaluate(self, particle):
        """Evaluates the particle (ANN parameters) using the loss function.

        Args:
            particle: The encoded particle representing ANN's parameters.

        Returns:
            Tuple: A tuple containing the accuracy and the loss value.
        """
        self.set_variable(particle)  # Set the ANN variables based on the particle
        acc, loss = self.ann.evaluate(self.x, self.y, self.loss)  # Evaluate the ANN
        return acc, loss  # Return accuracy and loss
