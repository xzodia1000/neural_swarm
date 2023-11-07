import numpy as np
from neural_swarm.activation.logistic import Logistic
from neural_swarm.activation.relu import Relu
from neural_swarm.activation.tanh import Tanh
from neural_swarm.ann.ann import ANN


class Particle:
    def __init__(self, ann):
        self._activation_functions_enc = {"Logistic": 0, "Relu": 1, "Tanh": 2}
        self._activation_functions_dec = {0: Logistic(), 1: Relu(), 2: Tanh()}

        self.ann = ann
        self.position = None
        self.pbest = None
        self.velocity = None
        self.fitness = None
        self.informants = None

    def encode(self):
        encoded = []
        for layer in self.ann.network.layers:
            if layer.weights is not None:
                encoded.extend(np.array(layer.weights).flatten())

        for layer in self.ann.network.layers:
            if layer.weights is not None:
                encoded.append(
                    self._activation_functions_enc[layer.activation.__str__()]
                )

        return encoded

    def decode(self, encoded):
        decoded = []
        encoded = np.array(encoded)
        for layer in self.ann.network.layers:
            if layer.weights is not None:
                weights = encoded[: layer.weights.size].reshape(layer.weights.shape)
                layer.update_weights(weights)
                decoded.append(weights)
                encoded = encoded[layer.weights.size :]

        for layer in self.ann.network.layers:
            if layer.weights is not None:
                activation = self._activation_functions_dec[encoded[0]]
                layer.update_activation(activation)
                decoded.append(activation)
                encoded = encoded[1:]

        return decoded
