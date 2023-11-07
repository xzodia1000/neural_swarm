import numpy as np
from neural_swarm.activation.activation import Activation


class Tanh(Activation):
    def evaluate(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def __str__(self):
        return "Tanh"
