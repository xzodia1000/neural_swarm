import numpy as np
from neural_swarm.ann.activation import Identity, Sigmoid, Relu, Softmax, Tanh


class ANNFunction:
    def __init__(self, ann, x, y, loss, optimize_activation=False):
        self.ann = ann
        self.x = x
        self.y = y
        self.loss = loss
        self.optimize_activation = optimize_activation

        self._activation_functions_dec = {
            0: Sigmoid(),
            1: Relu(),
            2: Tanh(),
            3: Softmax(),
            4: Identity(),
        }

    def get_ann(self):
        return self.ann

    def get_dimension(self):
        dimension = 0
        for layer in self.ann.network.layers:
            if layer.weights is not None:
                dimension += (
                    np.array(layer.weights).flatten().shape[0] + 1
                    if self.optimize_activation
                    else np.array(layer.weights).flatten().shape[0]
                )

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

                if self.optimize_activation:
                    activation_index = int(np.clip(np.floor(encoded[0] * 5), 0, 4))
                    activation_function = self._activation_functions_dec[
                        activation_index
                    ]
                    layer.set_activation(activation_function)
                    decoded.append(activation_function)
                    encoded = encoded[1:]

        return decoded

    def evaluate(self, variable):
        self.set_variable(variable)
        acc, loss = self.ann.evaluate(self.x, self.y, self.loss)
        return acc, loss
