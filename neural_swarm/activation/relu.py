import numpy as np
from neural_swarm.activation.activation import Activation


class Relu(Activation):
    def evaluate(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)
