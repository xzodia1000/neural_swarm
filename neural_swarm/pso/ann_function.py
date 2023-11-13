import numpy as np
from neural_swarm.ann.activation import Sigmoid, Relu, Tanh


class ANNFunction:
    def __init__(self, ann, x, y, loss):
        self.ann = ann
        self.x = x
        self.y = y
        self.loss = loss

        self._activation_functions_enc = {"Sigmoid": 0, "Relu": 1, "Tanh": 2}
        self._activation_functions_dec = {0: Sigmoid(), 1: Relu(), 2: Tanh()}

    def get_ann(self):
        return self.ann

    def get_dimension(self):
        dimension = 0
        for layer in self.ann.network.layers:
            if layer.weights is not None:
                dimension += np.array(layer.weights).flatten().shape[0]

        # for layer in self.fun.network.layers:
        #     if layer.weights is not None:
        #         encoded.append(
        #             self._activation_functions_enc[layer.activation.__str__()]
        #         )

        return dimension

    def set_variable(self, variable):
        decoded = []
        encoded = np.array(variable)
        for layer in self.ann.network.layers:
            if layer.weights is not None:
                weights = encoded[: layer.weights.size].reshape(layer.weights.shape)
                layer.set_weights(weights)
                decoded.append(weights)
                encoded = encoded[layer.weights.size :]

        # for layer in self.ann.network.layers:
        #     if layer.weights is not None:
        #         activation = self._activation_functions_dec[encoded[0]]
        #         layer.set_activation(activation)
        #         decoded.append(activation)
        #         encoded = encoded[1:]

        return decoded

    def evaluate(self, variable):
        self.set_variable(variable)
        acc, loss = self.ann.evaluate(self.x, self.y, self.loss)
        return acc, loss
